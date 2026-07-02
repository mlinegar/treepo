"""Setup helpers for HLL, Markov, and LDA examples."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from .common import artifact_kind, metric, neural_operator_backend
from .configs import HllSketchConfig, NeuralOperatorLDALeafGridConfig, NeuralOperatorMarkovLeafGridConfig


def hll_trees(config: HllSketchConfig) -> Sequence[Any]:
    from treepo.methods.fixtures import make_hll_item_trees

    return make_hll_item_trees(
        n_trees=int(config.n_trees),
        leaves_per_tree=int(config.leaves_per_tree),
        leaf_unit_count=int(config.leaf_unit_count),
        doc_unit_kind=str(config.doc_unit_kind),
        vocabulary_size=int(config.vocabulary_size),
        seed=int(config.seed),
    )


def hll_fit_config(config: HllSketchConfig, *, output_dir: Path, trees: Sequence[Any]) -> dict[str, Any]:
    return {
        "family": "classical_sketch",
        "train_data": trees,
        "eval_data": trees,
        "backend_config": {
            "output_dir": str(output_dir / "fit"),
            "sketch": "hll",
            "backend": str(config.backend),
            "precision": int(config.precision),
            "hash_bits": int(config.hash_bits),
            "schedule": str(config.schedule),
        },
        "axis": {
            "max_iterations": int(config.max_iterations),
            "axis_kind": "leaf_unit_count",
            "axis_value": int(config.leaf_unit_count),
        },
    }


def markov_split(config: Any, *, leaf_unit_count: int | None = None) -> tuple[Sequence[Any], Sequence[Any]]:
    from treepo.methods.fixtures import make_markov_changepoint_trees

    leaf_count = int(leaf_unit_count if leaf_unit_count is not None else config.leaf_unit_count)
    fixture_device = str(getattr(config, "fixture_device", None) or config.device)
    common = {
        "n_states": int(getattr(config, "n_states", 4)),
        "doc_tokens": int(config.doc_tokens),
        "leaf_unit_count": leaf_count,
        "doc_unit_kind": str(config.doc_unit_kind),
        "transition_prob": float(getattr(config, "transition_prob", 0.15)),
        "vocabulary_size": int(config.vocabulary_size),
        "generation_device": fixture_device,
    }
    return (
        make_markov_changepoint_trees(
            n_trees=int(config.n_train),
            seed=int(config.seed),
            split="train",
            **common,
        ),
        make_markov_changepoint_trees(
            n_trees=int(config.n_eval),
            seed=int(config.seed) + 1,
            split="test",
            **common,
        ),
    )


def validate_preference_mode(mode: str) -> None:
    if mode not in {"none", "scores", "pairwise", "ranked"}:
        raise ValueError("preference_mode must be one of: none, scores, pairwise, ranked")


def markov_preferences(trees: Sequence[Any], *, mode: str) -> Any | None:
    validate_preference_mode(mode)
    if mode == "none":
        return None

    from treepo import Candidate, PreferenceDataset, PreferenceRecord

    dataset = PreferenceDataset()
    for tree_idx, tree in enumerate(trees):
        meta = dict(getattr(tree, "metadata", None) or {})
        tree_id = str(meta.get("tree_id") or f"markov_{tree_idx}")
        root_label = float(meta.get("teacher_score_native", 0.0) or 0.0)
        dataset.append(
            PreferenceRecord(
                record_id=f"{tree_id}:root:{mode}",
                unit_id=f"{tree_id}:root",
                unit_type="root",
                target="f",
                context={"prompt": "Estimate the document-level Markov changepoint count."},
                candidates=(
                    Candidate(
                        id="oracle_root_count",
                        value=root_label,
                        score=1.0,
                        rank=1 if mode == "ranked" else None,
                        preferred=mode == "pairwise",
                    ),
                    Candidate(
                        id="under_count",
                        value=max(0.0, root_label - 1.0),
                        score=0.5,
                        rank=2 if mode == "ranked" else None,
                    ),
                ),
                metadata={"law_type": "markov_root_count", "preference_mode": mode},
            )
        )
        leaves = list(getattr(tree, "leaves", ()) or ())
        if leaves:
            dataset.append(_markov_leaf_preference(tree_id, leaves[0], mode=mode))
    return dataset


def _markov_leaf_preference(tree_id: str, leaf: Any, *, mode: str) -> Any:
    from treepo import Candidate, PreferenceRecord

    regimes = list(getattr(leaf, "regimes", ()) or ())
    tokens = list(getattr(leaf, "tokens", ()) or ())
    label = float(sum(1 for left, right in zip(regimes, regimes[1:]) if int(left) != int(right)))
    unit_id = f"{tree_id}:leaf:0"
    candidates = [
        Candidate(
            id="oracle_leaf_count",
            value=label,
            score=1.0,
            rank=1 if mode == "ranked" else None,
            preferred=mode == "pairwise",
        ),
        Candidate(
            id="under_count",
            value=max(0.0, label - 1.0),
            score=0.5,
            rank=2 if mode == "ranked" else None,
        ),
    ]
    if mode == "ranked":
        candidates.append(Candidate(id="empty_count", value=0.0, score=0.0, rank=3))
    return PreferenceRecord(
        record_id=f"{unit_id}:{mode}",
        unit_id=unit_id,
        unit_type="leaf",
        target="g",
        context={
            "prompt": "Estimate the leaf-level Markov changepoint count.",
            "leaf": f"tokens={tokens} regimes={regimes}",
        },
        candidates=tuple(candidates),
        weight=float(max(1, len(tokens))),
        propensity=1.0,
        metadata={"law_type": "markov_leaf_count", "preference_mode": mode},
    )


def markov_fit_config(
    config: Any,
    *,
    output_dir: Path,
    train: Sequence[Any],
    eval_trees: Sequence[Any],
    operator_kind: str,
    leaf_unit_count: int,
) -> dict[str, Any]:
    return {
        "family": "neural_operator",
        "train_data": train,
        "eval_data": eval_trees,
        "backend_config": neural_operator_backend(
            config,
            output_dir=output_dir / "fit",
            operator_kind=operator_kind,
        ),
        "axis": {
            "max_iterations": int(config.max_iterations),
            "axis_kind": "leaf_unit_count",
            "axis_value": int(leaf_unit_count),
        },
    }


def markov_average_guess_baseline(train: Sequence[object], eval_trees: Sequence[object]) -> dict[str, float]:
    train_scores = [_score(tree) for tree in train]
    eval_scores = [_score(tree) for tree in eval_trees]
    mean = sum(train_scores) / len(train_scores)
    mae = sum(abs(mean - value) for value in eval_scores) / len(eval_scores)
    return {"target": float(mean), "mae": float(mae)}


def markov_grid_row(
    *,
    config: NeuralOperatorMarkovLeafGridConfig,
    result: Any,
    leaf_unit_count: int,
    operator_kind: str,
    average_baseline: dict[str, float],
) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    artifacts = dict(result.artifacts or {})
    f_artifact = dict(artifacts.get("f") or {})
    g_artifact = dict(artifacts.get("g") or {})
    statistic = dict(artifacts.get("statistic") or {})
    statistic_info = dict(statistic.get("info") or {})
    return {
        "doc_unit_kind": str(config.doc_unit_kind),
        "leaf_unit_count": int(leaf_unit_count),
        "leaves_per_doc": int(math.ceil(int(config.doc_tokens) / int(leaf_unit_count))),
        "operator_kind": operator_kind,
        "status": result.status,
        "internal_f_mae": metric(metrics, "internal_f_mae"),
        "internal_f_pearson": metric(metrics, "internal_f_pearson"),
        "mean_prediction": metric(metrics, "mean_prediction"),
        "mean_teacher": metric(metrics, "mean_teacher"),
        "n": metric(metrics, "n"),
        "average_guess_target": float(average_baseline["target"]),
        "average_guess_mae": float(average_baseline["mae"]),
        "f_loss": metric(f_artifact, "loss"),
        "g_loss": metric(g_artifact, "loss"),
        "g_trained": g_artifact.get("trained"),
        "normalize_targets": g_artifact.get("normalize_targets"),
        "numeric_transition_state_weight": g_artifact.get("numeric_transition_state_weight"),
        "statistic_state_kind": statistic_info.get("state_kind"),
        "statistic_local_law_rows": metric(statistic, "local_law_row_count"),
        "artifact_kind": artifact_kind(result),
        "manifest_path": result.manifest_path,
    }


def lda_split(config: Any, *, leaf_unit_count: int | None = None) -> tuple[Sequence[Any], Sequence[Any]]:
    from treepo.methods.fixtures import make_lda_topic_trees

    leaf_count = int(leaf_unit_count if leaf_unit_count is not None else config.leaf_unit_count)
    fixture_device = str(getattr(config, "fixture_device", None) or config.device)
    common = {
        "n_topics": int(config.n_topics),
        "doc_tokens": int(config.doc_tokens),
        "leaf_unit_count": leaf_count,
        "doc_unit_kind": str(config.doc_unit_kind),
        "vocabulary_size": int(config.vocabulary_size),
        "doc_topic_concentration": float(config.doc_topic_concentration),
        "topic_word_concentration": float(config.topic_word_concentration),
        "target_topic": int(config.target_topic),
        "topic_seed": int(config.topic_seed),
        "generation_device": fixture_device,
    }
    return (
        make_lda_topic_trees(
            n_trees=int(config.n_train),
            seed=int(config.seed),
            split="train",
            **common,
        ),
        make_lda_topic_trees(
            n_trees=int(config.n_eval),
            seed=int(config.seed) + 1,
            split="test",
            **common,
        ),
    )


def lda_target_key(config: Any) -> str:
    return f"topic_{int(config.target_topic)}_proportion"


def lda_fit_config(
    config: Any,
    *,
    output_dir: Path,
    train: Sequence[Any],
    eval_trees: Sequence[Any],
    operator_kind: str,
    leaf_unit_count: int,
) -> dict[str, Any]:
    backend = neural_operator_backend(config, output_dir=output_dir / "fit", operator_kind=operator_kind)
    backend.update(
        {
            "target_key": lda_target_key(config),
            "target_vector_key": "topic_proportions",
            "target_dim": int(config.n_topics),
            "target_min": 0.0,
            "target_max": 1.0,
        }
    )
    return {
        "family": "neural_operator",
        "train_data": train,
        "eval_data": eval_trees,
        "backend_config": backend,
        "axis": {
            "max_iterations": int(config.max_iterations),
            "axis_kind": "leaf_unit_count",
            "axis_value": int(leaf_unit_count),
        },
    }


def fit_lda_sklearn_baseline(config: Any, train: Sequence[Any], eval_trees: Sequence[Any]) -> Any | None:
    if not bool(config.run_sklearn_baseline) or int(config.sklearn_max_iter) <= 0:
        return None

    from treepo.methods.lda import fit_sklearn_lda_baseline

    return fit_sklearn_lda_baseline(
        train,
        eval_trees,
        n_topics=int(config.n_topics),
        vocabulary_size=int(config.vocabulary_size),
        doc_topic_prior=float(config.doc_topic_concentration),
        topic_word_prior=float(config.topic_word_concentration),
        max_iter=int(config.sklearn_max_iter),
        random_state=int(config.seed),
        target_topic=int(config.target_topic),
    )


def lda_average_guess_baseline(
    train: Sequence[object],
    eval_trees: Sequence[object],
    n_topics: int,
    target_topic: int,
) -> dict[str, float]:
    train_vectors = [_topic_vector(tree, n_topics) for tree in train]
    eval_vectors = [_topic_vector(tree, n_topics) for tree in eval_trees]
    means = [sum(row[idx] for row in train_vectors) / len(train_vectors) for idx in range(n_topics)]
    errors = [[abs(means[idx] - row[idx]) for idx in range(n_topics)] for row in eval_vectors]
    return {
        "target": float(means[target_topic]),
        "target_mae": float(sum(row[target_topic] for row in errors) / len(errors)),
        "mean_mae": float(sum(sum(row) for row in errors) / (len(errors) * n_topics)),
    }


def _topic_vector(tree: object, n_topics: int) -> list[float]:
    values = getattr(tree, "topic_proportions", None)
    if values is None:
        values = getattr(tree, "metadata", {}).get("topic_proportions")
    out = [float(value) for value in values]
    if len(out) != int(n_topics):
        raise ValueError(f"expected {n_topics} topic proportions, got {len(out)}")
    return out


def lda_grid_row(
    *,
    config: NeuralOperatorLDALeafGridConfig,
    result: Any,
    leaf_unit_count: int,
    operator_kind: str,
    sklearn_baseline: Any | None,
    average_baseline: dict[str, float],
) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    artifacts = dict(result.artifacts or {})
    statistic = dict(artifacts.get("statistic") or {})
    statistic_info = dict(statistic.get("info") or {})
    return {
        "doc_unit_kind": str(config.doc_unit_kind),
        "leaf_unit_count": int(leaf_unit_count),
        "leaves_per_doc": int(math.ceil(int(config.doc_tokens) / int(leaf_unit_count))),
        "operator_kind": operator_kind,
        "status": result.status,
        "internal_f_mae": metric(metrics, "internal_f_mae"),
        "internal_f_pearson": metric(metrics, "internal_f_pearson"),
        "target_topic_vector_mae": metric(metrics, f"topic_{int(config.target_topic)}_internal_f_mae"),
        "mean_topic_vector_mae": mean_topic_mae(metrics, int(config.n_topics)),
        "mean_prediction": metric(metrics, "mean_prediction"),
        "mean_teacher": metric(metrics, "mean_teacher"),
        "n": metric(metrics, "n"),
        "sklearn_target_mae": None if sklearn_baseline is None else float(sklearn_baseline.target_mae),
        "sklearn_mean_mae": None if sklearn_baseline is None else float(sklearn_baseline.mean_mae),
        "average_guess_target": float(average_baseline["target"]),
        "average_guess_target_mae": float(average_baseline["target_mae"]),
        "average_guess_mean_mae": float(average_baseline["mean_mae"]),
        "statistic_state_kind": statistic_info.get("state_kind"),
        "statistic_local_law_rows": metric(statistic, "local_law_row_count"),
        "artifact_kind": artifact_kind(result),
        "manifest_path": result.manifest_path,
    }


def mean_topic_mae(metrics: dict[str, Any], n_topics: int) -> float | None:
    values = [metric(metrics, f"topic_{idx}_internal_f_mae") for idx in range(max(1, int(n_topics)))]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _score(tree: object) -> float:
    return float(getattr(tree, "metadata", {}).get("teacher_score_native"))

