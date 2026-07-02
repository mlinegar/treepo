"""Split evaluation and metric helpers for methods runtimes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from treepo.methods._runtime_types import SplitMetrics
from treepo.methods.contracts import FamilyRuntime


def evaluate_splits(
    family: FamilyRuntime,
    f_artifact: Any,
    g_artifact: Any,
    trees: Sequence[Any],
    *,
    prediction_rows: list[dict[str, Any]] | None = None,
) -> dict[str, SplitMetrics]:
    tree_list = list(trees or [])
    if not tree_list:
        return {}
    predictions = family.score_roots_with_f(f=f_artifact, g=g_artifact, trees=tree_list)
    if prediction_rows is not None:
        prediction_rows.extend(_prediction_rows(predictions, tree_list))
    groups: dict[str, list[int]] = {"all": list(range(len(tree_list)))}
    for idx, tree in enumerate(tree_list):
        split = _tree_split(tree)
        groups.setdefault(split, []).append(idx)
    out: dict[str, SplitMetrics] = {}
    for split, indices in groups.items():
        split_preds = [predictions[i] if i < len(predictions) else None for i in indices]
        split_trees = [tree_list[i] for i in indices]
        out[split] = _split_metrics(split_preds, split_trees)
    return out


def _prediction_rows(
    predictions: Sequence[Any | None],
    trees: Sequence[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, tree in enumerate(trees):
        prediction = predictions[idx] if idx < len(predictions) else None
        rows.append(
            {
                "tree_index": int(idx),
                "tree_id": _tree_id(tree, idx),
                "split": _tree_split(tree),
                "prediction": _json_prediction(prediction),
                "prediction_scalar": _prediction_scalar(prediction, tree),
                "teacher_score": _teacher_root_score(tree),
                "teacher_vector": _truth_vector(tree),
                "expert_score": _expert_root_score(tree),
            }
        )
    return rows


def _split_metrics(predictions: Sequence[Any | None], trees: Sequence[Any]) -> SplitMetrics:
    scalar_predictions = [_prediction_scalar(pred, tree) for pred, tree in zip(predictions, trees)]
    teacher = [_teacher_root_score(tree) for tree in trees]
    expert = [_expert_root_score(tree) for tree in trees]
    internal = _paired_stats(scalar_predictions, teacher)
    external = _paired_stats(scalar_predictions, expert)
    per_dimension = _per_dimension_metrics(predictions, trees)
    gap = None
    if internal["pearson"] is not None and external["pearson"] is not None:
        gap = float(internal["pearson"] - external["pearson"])
    return SplitMetrics(
        n=int(max(internal["n"], external["n"])),
        internal_f_pearson=internal["pearson"],
        internal_f_mae=internal["mae"],
        external_expert_pearson=external["pearson"],
        external_expert_mae=external["mae"],
        f_star_gap=gap,
        mean_prediction=internal["mean_prediction"] or external["mean_prediction"],
        mean_teacher=internal["mean_truth"],
        mean_expert=external["mean_truth"],
        metrics_scale=_metrics_scale(trees),
        per_dimension=per_dimension,
    )


def _paired_stats(
    predictions: Sequence[float | None],
    truths: Sequence[float | None],
) -> dict[str, float | None]:
    paired = []
    for p, t in zip(predictions, truths):
        pred = _safe_float(p)
        truth = _safe_float(t)
        if pred is not None and truth is not None:
            paired.append((pred, truth))
    if not paired:
        return {
            "n": 0,
            "pearson": None,
            "mae": None,
            "mean_prediction": None,
            "mean_truth": None,
        }
    ps, ts = zip(*paired)
    mae = sum(abs(p - t) for p, t in paired) / len(paired)
    return {
        "n": float(len(paired)),
        "pearson": _pearson(ps, ts),
        "mae": float(mae),
        "mean_prediction": float(sum(ps) / len(ps)),
        "mean_truth": float(sum(ts) / len(ts)),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return float(num / (den_x * den_y))


def _json_prediction(value: Any) -> Any:
    vector = _as_float_vector(value)
    if vector is not None:
        return vector
    return _safe_float(value)


def _prediction_scalar(value: Any, tree: Any) -> float | None:
    vector = _as_float_vector(value)
    if vector is None:
        return _safe_float(value)
    target_topic = int(_metadata(tree).get("target_topic", 0) or 0)
    if 0 <= target_topic < len(vector):
        return float(vector[target_topic])
    return float(vector[0]) if vector else None


def _as_float_vector(value: Any) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        out = [float(item) for item in value]
    except TypeError:
        return None
    return out if out else None


def _truth_vector(tree: Any) -> list[float] | None:
    meta = _metadata(tree)
    value = getattr(tree, "topic_proportions", None)
    if value is None:
        value = meta.get("topic_proportions")
    return _as_float_vector(value)


def _per_dimension_metrics(
    predictions: Sequence[Any | None],
    trees: Sequence[Any],
) -> dict[str, dict[str, float | None]]:
    pred_vectors = [_as_float_vector(pred) for pred in predictions]
    truth_vectors = [_truth_vector(tree) for tree in trees]
    widths = [len(vec) for vec in pred_vectors if vec is not None]
    truth_widths = [len(vec) for vec in truth_vectors if vec is not None]
    if not widths or not truth_widths:
        return {}
    width = min(min(widths), min(truth_widths))
    out: dict[str, dict[str, float | None]] = {}
    for dim in range(width):
        dim_preds = [
            vec[dim] if vec is not None and len(vec) > dim else None
            for vec in pred_vectors
        ]
        dim_truths = [
            vec[dim] if vec is not None and len(vec) > dim else None
            for vec in truth_vectors
        ]
        stats = _paired_stats(dim_preds, dim_truths)
        out[f"topic_{dim}"] = {
            "internal_f_pearson": stats["pearson"],
            "internal_f_mae": stats["mae"],
            "mean_prediction": stats["mean_prediction"],
            "mean_teacher": stats["mean_truth"],
            "n": stats["n"],
        }
    return out


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metadata(tree: Any) -> Mapping[str, Any]:
    meta = getattr(tree, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def _tree_split(tree: Any) -> str:
    return str(_metadata(tree).get("split") or "unknown").lower()


def _tree_id(tree: Any, fallback: int) -> str:
    meta = _metadata(tree)
    value = (
        getattr(tree, "doc_id", None)
        or getattr(tree, "tree_id", None)
        or meta.get("doc_id")
        or meta.get("tree_id")
    )
    return str(value) if value is not None else str(fallback)


def _teacher_root_score(tree: Any) -> float | None:
    meta = _metadata(tree)
    for key in ("teacher_score_native", "teacher_score_1_7", "document_score"):
        value = meta.get(key) if key != "document_score" else getattr(tree, key, None)
        score = _safe_float(value)
        if score is not None:
            return score
    return None


def _expert_root_score(tree: Any) -> float | None:
    meta = _metadata(tree)
    for key in (
        "expert_score_native",
        "expert_score_for_objective",
        "expert_score_1_7",
        "teacher_score_native",
        "teacher_score_1_7",
    ):
        score = _safe_float(meta.get(key))
        if score is not None:
            return score
    return _safe_float(getattr(tree, "document_score", None))


def _metrics_scale(trees: Sequence[Any]) -> str:
    for tree in trees:
        value = _metadata(tree).get("expert_target_scale")
        if value:
            return str(value)
    return "native"


__all__ = ["evaluate_splits"]
