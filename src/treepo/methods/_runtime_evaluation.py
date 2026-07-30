"""Split evaluation and metric helpers for methods runtimes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from treepo.forest import oracle_vector_l1
from treepo.methods._coerce import float_vector as _as_float_vector
from treepo.methods._coerce import safe_float
from treepo.methods._runtime_types import SplitMetrics
from treepo.methods.contracts import FamilyRuntime
from treepo.tree import tree_root_target, tree_row_id


def evaluate_splits(
    family: FamilyRuntime,
    f_artifact: Any,
    g_artifact: Any,
    trees: Sequence[Any],
    *,
    prediction_rows: list[dict[str, Any]] | None = None,
    reporting_oracle_targets: Sequence[Any] = (),
) -> dict[str, SplitMetrics]:
    tree_list = list(trees or [])
    if not tree_list:
        return {}
    config = getattr(family, "config", None)
    target_names = tuple(getattr(config, "target_names", ()) or ())
    target_oracle_ids = tuple(getattr(config, "target_oracle_ids", ()) or ())
    target_key = getattr(config, "target_key", None)
    target_vector_key = getattr(config, "target_vector_key", None)
    reporting_target_names, reporting_target_oracle_ids = _reporting_schema(
        reporting_oracle_targets,
        fallback_names=target_names,
        fallback_oracle_ids=target_oracle_ids,
    )
    predictions = family.score_roots_with_f(f=f_artifact, g=g_artifact, trees=tree_list)
    if prediction_rows is not None:
        prediction_rows.extend(
            _prediction_rows(
                predictions,
                tree_list,
                target_names=target_names,
                target_oracle_ids=target_oracle_ids,
                target_key=target_key,
                target_vector_key=target_vector_key,
                reporting_target_names=reporting_target_names,
                reporting_target_oracle_ids=reporting_target_oracle_ids,
            )
        )
    groups: dict[str, list[int]] = {"all": list(range(len(tree_list)))}
    for idx, tree in enumerate(tree_list):
        split = _tree_split(tree)
        groups.setdefault(split, []).append(idx)
    out: dict[str, SplitMetrics] = {}
    for split, indices in groups.items():
        split_preds = [predictions[i] if i < len(predictions) else None for i in indices]
        split_trees = [tree_list[i] for i in indices]
        out[split] = _split_metrics(
            split_preds,
            split_trees,
            target_names=target_names,
            target_key=target_key,
            target_vector_key=target_vector_key,
            reporting_target_names=reporting_target_names,
        )
    return out


def _prediction_rows(
    predictions: Sequence[Any | None],
    trees: Sequence[Any],
    *,
    target_names: Sequence[str] = (),
    target_oracle_ids: Sequence[str] = (),
    target_key: str | None = None,
    target_vector_key: str | None = None,
    reporting_target_names: Sequence[str] | None = None,
    reporting_target_oracle_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    report_names = tuple(target_names if reporting_target_names is None else reporting_target_names)
    report_oracle_ids = tuple(
        target_oracle_ids if reporting_target_oracle_ids is None else reporting_target_oracle_ids
    )
    legacy_target_names = (
        () if reporting_target_names is not None and len(report_names) == 1 else tuple(target_names)
    )
    rows: list[dict[str, Any]] = []
    for idx, tree in enumerate(trees):
        prediction = predictions[idx] if idx < len(predictions) else None
        prediction_scalar = _prediction_scalar(
            prediction,
            tree,
            target_names=tuple(target_names),
        )
        legacy_truth_vector = _truth_vector(
            tree,
            target_names=legacy_target_names,
            target_key=target_key,
            target_vector_key=target_vector_key,
        )
        teacher_score = _teacher_root_score(tree, target_names=legacy_target_names)
        expert_score = _expert_root_score(tree, target_names=legacy_target_names)
        row = {
            "tree_index": int(idx),
            "tree_id": _tree_id(tree, idx),
            "split": _tree_split(tree),
            # Preserve the legacy scalar compatibility column for a named
            # singleton. The native execution value remains a one-coordinate
            # vector and is recorded without projection in
            # prediction_by_target below.
            "prediction": (
                prediction_scalar
                if len(report_names) == 1 and bool(target_names)
                else _json_prediction(prediction)
            ),
            "prediction_scalar": prediction_scalar,
            "teacher_score": teacher_score,
            "teacher_vector": legacy_truth_vector,
            "expert_score": expert_score,
        }
        if report_names:
            prediction_vector = _named_vector(
                prediction,
                report_names,
                source="model prediction",
            )
            reporting_truth_vector = _truth_vector(
                tree,
                target_names=report_names,
                target_key=target_key,
                target_vector_key=target_vector_key,
            )
            row.update(
                {
                    "target_order": list(report_names),
                    "oracle_ids_by_target": dict(zip(report_names, report_oracle_ids)),
                    "prediction_by_target": (
                        dict(zip(report_names, prediction_vector))
                        if prediction_vector is not None
                        else None
                    ),
                    "target_by_name": (
                        dict(zip(report_names, reporting_truth_vector))
                        if reporting_truth_vector is not None
                        else None
                    ),
                }
            )
        rows.append(row)
    return rows


def _split_metrics(
    predictions: Sequence[Any | None],
    trees: Sequence[Any],
    *,
    target_names: Sequence[str] = (),
    target_key: str | None = None,
    target_vector_key: str | None = None,
    reporting_target_names: Sequence[str] | None = None,
) -> SplitMetrics:
    report_names = tuple(target_names if reporting_target_names is None else reporting_target_names)
    legacy_target_names = (
        () if reporting_target_names is not None and len(report_names) == 1 else tuple(target_names)
    )
    scalar_predictions = [
        _prediction_scalar(pred, tree, target_names=tuple(target_names))
        for pred, tree in zip(predictions, trees)
    ]
    teacher = [_teacher_root_score(tree, target_names=legacy_target_names) for tree in trees]
    expert = [_expert_root_score(tree, target_names=legacy_target_names) for tree in trees]
    internal = _paired_stats(scalar_predictions, teacher)
    external = _paired_stats(scalar_predictions, expert)
    per_dimension = _per_dimension_metrics(
        predictions,
        trees,
        target_names=report_names,
        target_key=target_key,
        target_vector_key=target_vector_key,
    )
    joint_l1 = _joint_l1_metrics(
        predictions,
        trees,
        target_names=report_names,
        target_key=target_key,
        target_vector_key=target_vector_key,
        legacy_target_names=legacy_target_names,
    )
    gap = None
    if internal["pearson"] is not None and external["pearson"] is not None:
        gap = float(internal["pearson"] - external["pearson"])
    # The runtime supplies ``reporting_target_names`` for a declared catalog.
    # If that catalog is also native to the family, its dimension counts give
    # vector-only K=1 fits document coverage. A scalar-only singleton lift (or
    # the direct legacy helper path) retains the established scalar ``n``.
    native_named_catalog = (
        reporting_target_names is not None
        and bool(target_names)
        and tuple(target_names) == report_names
    )
    dimension_ns = (
        [_safe_float(metrics.get("n")) or 0.0 for metrics in per_dimension.values()]
        if len(report_names) != 1 or native_named_catalog
        else []
    )
    return SplitMetrics(
        n=int(max([internal["n"], external["n"], *dimension_ns])),
        joint_f_l1=joint_l1["mean"],
        joint_f_l1_n=int(joint_l1["n"]),
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


def _joint_l1_metrics(
    predictions: Sequence[Any | None],
    trees: Sequence[Any],
    *,
    target_names: Sequence[str],
    target_key: str | None,
    target_vector_key: str | None,
    legacy_target_names: Sequence[str],
) -> dict[str, float | int | None]:
    """Evaluate one native L1 point distance per observed tree.

    Named vectors use their declared order. Anonymous vector tasks retain
    their vector width. Anonymous scalars are lifted to one-coordinate vectors
    and compared to the established internal/teacher target.
    """

    distances: list[float] = []
    for prediction, tree in zip(predictions, trees):
        if target_names:
            predicted = _named_vector(
                prediction,
                target_names,
                source="model prediction",
            )
            observed = _truth_vector(
                tree,
                target_names=target_names,
                target_key=target_key,
                target_vector_key=target_vector_key,
            )
        else:
            predicted = _as_float_vector(prediction)
            observed = _truth_vector(
                tree,
                target_key=target_key,
                target_vector_key=target_vector_key,
            )
            if predicted is None or observed is None:
                scalar_prediction = _prediction_scalar(
                    prediction,
                    tree,
                    target_names=legacy_target_names,
                )
                scalar_target = _teacher_root_score(
                    tree,
                    target_names=legacy_target_names,
                )
                predicted = [scalar_prediction] if scalar_prediction is not None else None
                observed = [scalar_target] if scalar_target is not None else None
        if predicted is None or observed is None:
            continue
        distances.append(oracle_vector_l1(predicted, observed))
    return {
        "n": len(distances),
        "mean": (float(sum(distances) / len(distances)) if distances else None),
    }


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


def _prediction_scalar(
    value: Any,
    tree: Any,
    *,
    target_names: Sequence[str] = (),
) -> float | None:
    if target_names:
        # The singleton product is canonically the scalar target. Wider named
        # vectors have no declared scalarization and must never default to
        # coordinate zero.
        if len(target_names) != 1:
            return None
        vector = _named_vector(value, target_names, source="model prediction")
        return vector[0] if vector is not None else None
    vector = _as_float_vector(value)
    if vector is None:
        return _safe_float(value)
    target_topic = int(_metadata(tree).get("target_topic", 0) or 0)
    if 0 <= target_topic < len(vector):
        return float(vector[target_topic])
    return float(vector[0]) if vector else None


def _truth_vector(
    tree: Any,
    *,
    target_names: Sequence[str] = (),
    target_key: str | None = None,
    target_vector_key: str | None = None,
) -> list[float] | None:
    meta = _metadata(tree)
    if target_names:
        if target_vector_key:
            value = (
                meta.get(target_vector_key)
                if target_vector_key in meta
                else getattr(tree, target_vector_key, None)
            )
        else:
            if len(target_names) == 1:
                score = tree_root_target(tree, target_key=target_key)
                if score is not None:
                    return [float(score)]
            value = getattr(tree, "root_label", None)
            if value is None:
                value = getattr(tree, "document_score", None)
        return _named_vector(value, target_names, source="joint oracle target")
    value = getattr(tree, "topic_proportions", None)
    if value is None:
        value = meta.get("topic_proportions")
    return _as_float_vector(value)


def _per_dimension_metrics(
    predictions: Sequence[Any | None],
    trees: Sequence[Any],
    *,
    target_names: Sequence[str] = (),
    target_key: str | None = None,
    target_vector_key: str | None = None,
) -> dict[str, dict[str, float | None]]:
    if target_names:
        pred_vectors = [
            _named_vector(pred, target_names, source="model prediction") for pred in predictions
        ]
    else:
        pred_vectors = [_as_float_vector(pred) for pred in predictions]
    truth_vectors = [
        _truth_vector(
            tree,
            target_names=target_names,
            target_key=target_key,
            target_vector_key=target_vector_key,
        )
        for tree in trees
    ]
    widths = [len(vec) for vec in pred_vectors if vec is not None]
    truth_widths = [len(vec) for vec in truth_vectors if vec is not None]
    if not widths or not truth_widths:
        return {}
    width = min(min(widths), min(truth_widths))
    out: dict[str, dict[str, float | None]] = {}
    for dim in range(width):
        dim_preds = [
            vec[dim] if vec is not None and len(vec) > dim else None for vec in pred_vectors
        ]
        dim_truths = [
            vec[dim] if vec is not None and len(vec) > dim else None for vec in truth_vectors
        ]
        stats = _paired_stats(dim_preds, dim_truths)
        dimension_name = target_names[dim] if target_names else f"topic_{dim}"
        out[dimension_name] = {
            "internal_f_pearson": stats["pearson"],
            "internal_f_mae": stats["mae"],
            "mean_prediction": stats["mean_prediction"],
            "mean_teacher": stats["mean_truth"],
            "n": stats["n"],
        }
    return out


def _reporting_schema(
    reporting_oracle_targets: Sequence[Any],
    *,
    fallback_names: Sequence[str],
    fallback_oracle_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve the ordered reporting schema independently of family I/O."""

    targets = tuple(reporting_oracle_targets or ())
    if not targets:
        return tuple(fallback_names), tuple(fallback_oracle_ids)
    names: list[str] = []
    oracle_ids: list[str] = []
    for target in targets:
        if isinstance(target, Mapping):
            name = target.get("target_name")
            oracle_id = target.get("oracle_id")
        else:
            name = getattr(target, "target_name", None)
            oracle_id = getattr(target, "oracle_id", None)
        target_name = str(name or "").strip()
        target_oracle_id = str(oracle_id or "").strip()
        if not target_name or not target_oracle_id:
            raise ValueError(
                "reporting_oracle_targets must provide non-empty target_name and oracle_id"
            )
        names.append(target_name)
        oracle_ids.append(target_oracle_id)
    if len(names) != len(set(names)):
        raise ValueError("reporting_oracle_targets target_name values must be unique")
    return tuple(names), tuple(oracle_ids)


def _named_vector(
    value: Any,
    target_names: Sequence[str],
    *,
    source: str,
) -> list[float] | None:
    if value is None:
        return None
    names = tuple(str(name) for name in target_names)
    if isinstance(value, Mapping):
        by_name = {str(key): component for key, component in value.items()}
        if len(by_name) != len(value):
            raise ValueError(f"{source} contains keys that collide after string coercion")
        actual = set(by_name)
        expected = set(names)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{source} mapping must cover the declared target names exactly; "
                f"missing={missing}, extra={extra}"
            )
        vector = [_safe_float(by_name[name]) for name in names]
    else:
        raw_vector = _as_float_vector(value)
        vector = (
            [_safe_float(component) for component in raw_vector] if raw_vector is not None else None
        )
        if vector is None and len(names) == 1:
            scalar = _safe_float(value)
            vector = [scalar] if scalar is not None else None
    if vector is None:
        raise ValueError(f"{source} must be a finite vector aligned to target_names")
    if len(vector) != len(names):
        raise ValueError(
            f"{source} width {len(vector)} does not match the declared target width {len(names)}"
        )
    if any(component is None for component in vector):
        raise ValueError(f"{source} must contain only finite numeric values")
    return [float(component) for component in vector]


def _safe_float(value: Any) -> float | None:
    return safe_float(value, require_finite=True)


def _metadata(tree: Any) -> Mapping[str, Any]:
    meta = getattr(tree, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def _tree_split(tree: Any) -> str:
    return str(_metadata(tree).get("split") or "unknown").lower()


def _tree_id(tree: Any, fallback: int) -> str:
    return tree_row_id(tree, fallback, fallback_prefix=None)


def _teacher_root_score(
    tree: Any,
    *,
    target_names: Sequence[str] = (),
) -> float | None:
    meta = _metadata(tree)
    score = _safe_float(meta.get("teacher_score_native"))
    if score is not None:
        return score
    for field_name in ("document_score", "root_label"):
        value = getattr(tree, field_name, None)
        score = _safe_float(value)
        if score is not None:
            return score
        if len(target_names) == 1 and isinstance(value, Mapping):
            vector = _named_vector(
                value,
                target_names,
                source=f"internal {field_name}",
            )
            if vector is not None:
                return vector[0]
    return None


def _expert_root_score(
    tree: Any,
    *,
    target_names: Sequence[str] = (),
) -> float | None:
    # Deliberately NO fallback to teacher_score_native: external metrics must
    # measure agreement with a genuinely external label, otherwise f_star_gap
    # trivially reads 0 and internal/external silently alias.
    meta = _metadata(tree)
    score = _safe_float(meta.get("expert_score_for_objective"))
    if score is not None:
        return score
    for field_name in ("document_score", "root_label"):
        value = getattr(tree, field_name, None)
        score = _safe_float(value)
        if score is not None:
            return score
        if len(target_names) == 1 and isinstance(value, Mapping):
            vector = _named_vector(
                value,
                target_names,
                source=f"external {field_name}",
            )
            if vector is not None:
                return vector[0]
    return None


def _metrics_scale(trees: Sequence[Any]) -> str:
    for tree in trees:
        value = _metadata(tree).get("expert_target_scale")
        if value:
            return str(value)
    return "native"


__all__ = ["evaluate_splits"]
