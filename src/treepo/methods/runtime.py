"""Small alternating f/g runtime for the publishable methods surface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.methods.contracts import FamilyRuntime


def _stage_powers_for_iteration(k: int) -> tuple[int, int]:
    if k < 0:
        raise ValueError(f"iteration must be >= 0, got {k}")
    f_degree = 1
    g_degree = 1
    side = "f"
    for _ in range(int(k)):
        if side == "f":
            f_degree += 1
            side = "g"
        else:
            g_degree += 1
            side = "f"
    return f_degree, g_degree


def _stage_name_for_iteration(k: int) -> str:
    if k == 0:
        return "fg"
    return "fg" + "".join("f" if i % 2 == 0 else "g" for i in range(k))


def _stage_label_for_iteration(k: int) -> str:
    f_degree, g_degree = _stage_powers_for_iteration(k)
    return f"f^{f_degree} g^{g_degree}"


def _trains_f_at_iteration(k: int) -> bool:
    return k >= 1 and k % 2 == 1


def _trains_g_at_iteration(k: int) -> bool:
    return k >= 1 and k % 2 == 0


@dataclass
class SplitMetrics:
    n: int
    internal_f_pearson: float | None = None
    internal_f_mae: float | None = None
    internal_f_mae_1_7: float | None = None
    external_expert_pearson: float | None = None
    external_expert_mae: float | None = None
    external_expert_mae_1_7: float | None = None
    f_star_gap: float | None = None
    mean_prediction: float | None = None
    mean_teacher: float | None = None
    mean_expert: float | None = None
    mean_prediction_1_7: float | None = None
    mean_teacher_1_7: float | None = None
    mean_expert_1_7: float | None = None
    metrics_scale: str | None = None
    per_dimension: dict[str, dict[str, float | None]] = field(default_factory=dict)


@dataclass
class IterationRecord:
    iteration: int
    stage_name: str
    family: str
    trained: str
    stage_label: str | None = None
    f_degree: int | None = None
    g_degree: int | None = None
    axis_kind: str = "leaf_count"
    axis_value: int = 0
    leaf_count: int | None = None
    leaf_size_tokens: int | None = None
    f_artifact: Any = None
    g_artifact: Any = None
    split_metrics: dict[str, SplitMetrics] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def run_alternating_family(
    *,
    family: FamilyRuntime,
    f_init: Any,
    g_init: Any,
    traces: Sequence[Any],
    eval_trees: Sequence[Any],
    max_iterations: int,
    axis_value: int,
    output_dir: Path,
    axis_kind: str = "leaf_count",
    leaf_count: int | None = None,
    leaf_size_tokens: int | None = None,
) -> list[IterationRecord]:
    """Run a compact alternating loop over a public ``FamilyRuntime``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    f_artifact = f_init
    g_artifact = g_init
    records: list[IterationRecord] = []
    max_k = max(0, int(max_iterations))
    for k in range(max_k + 1):
        trained = "none"
        iteration_dir = output_dir / f"iter_{k:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        if _trains_f_at_iteration(k):
            f_artifact = family.train_f(
                f_init=f_artifact,
                g=g_artifact,
                traces=traces,
                output_dir=iteration_dir,
                iteration=k,
            )
            family.validate_artifact(kind="f", artifact=f_artifact)
            trained = "f"
        elif _trains_g_at_iteration(k):
            g_artifact = family.train_g(
                g_init=g_artifact,
                f=f_artifact,
                traces=traces,
                output_dir=iteration_dir,
                iteration=k,
            )
            family.validate_artifact(kind="g", artifact=g_artifact)
            trained = "g"
        f_degree, g_degree = _stage_powers_for_iteration(k)
        prediction_rows: list[dict[str, Any]] = []
        split_metrics = evaluate_splits(
            family,
            f_artifact,
            g_artifact,
            eval_trees,
            prediction_rows=prediction_rows,
        )
        records.append(
            IterationRecord(
                iteration=k,
                stage_name=_stage_name_for_iteration(k),
                stage_label=_stage_label_for_iteration(k),
                family=str(getattr(family, "name", type(family).__name__)),
                trained=trained,
                f_degree=f_degree,
                g_degree=g_degree,
                axis_kind=str(axis_kind),
                axis_value=int(axis_value),
                leaf_count=leaf_count,
                leaf_size_tokens=leaf_size_tokens,
                f_artifact=f_artifact,
                g_artifact=g_artifact,
                split_metrics=split_metrics,
                extra={"prediction_rows": prediction_rows} if prediction_rows else {},
            )
        )
    return records


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
        internal_f_mae_1_7=internal["mae"],
        external_expert_pearson=external["pearson"],
        external_expert_mae=external["mae"],
        external_expert_mae_1_7=external["mae"],
        f_star_gap=gap,
        mean_prediction=internal["mean_prediction"] or external["mean_prediction"],
        mean_teacher=internal["mean_truth"],
        mean_expert=external["mean_truth"],
        mean_prediction_1_7=internal["mean_prediction"] or external["mean_prediction"],
        mean_teacher_1_7=internal["mean_truth"],
        mean_expert_1_7=external["mean_truth"],
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
        return {"n": 0, "pearson": None, "mae": None, "mean_prediction": None, "mean_truth": None}
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
        dim_preds = [vec[dim] if vec is not None and len(vec) > dim else None for vec in pred_vectors]
        dim_truths = [vec[dim] if vec is not None and len(vec) > dim else None for vec in truth_vectors]
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


__all__ = [
    "IterationRecord",
    "SplitMetrics",
    "evaluate_splits",
    "run_alternating_family",
]
