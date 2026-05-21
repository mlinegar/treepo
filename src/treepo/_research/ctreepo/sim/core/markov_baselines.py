"""Classical baseline fitters extracted from markov_changepoint_ops_count.py.

Contains ridge regression, random forest, decision tree, KNN, endpoint table,
transformer, and doc-level baselines. All private functions — imported by the
main module's run function and re-exported for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from treepo._research.training.config_sections import OptimizerConfig, RunConfig, RuntimeConfig, TrainConfig
from treepo._research.training.supervision import (
    DenseScalarModelConfig,
    DenseScalarObjectiveConfig,
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseScalarTrainingConfig,
    SupervisionDataset,
    fit_dense_scalar_regressor,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_regressor,
    predict_dense_scalar_ridge_regressor,
)
from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    AdditiveCountSketch,
    FullSequenceBoundaryTransformer,
    FullSequenceCTreePOOperator,
    OPSCountConfig,
    SketchMetrics,
    TrainFitDiagnostics,
    _CountDoc,
    _SampledLeafPoolDoc,
    _dense_doc_matrix_supervision_dataset,
    _doc_level_feature_matrix,
    _doc_level_supervision_dataset,
    _doc_root_targets,
    _doc_token_ngram_feature_matrix,
    _eval_doc_level_dense_predictions,
    _eval_learned_model,
    _eval_root_only_predictions,
    _eval_root_predictions,
    _exact_match_rate,
    _leaf_core_feature_vector,
    _leaf_endpoint_table_feature_vector,
    _leaf_endpoint_table_key,
    _ngram_order_label,
    _normalized_ngram_orders,
    _sampled_leaf_pool_feature_matrix,
    _token_sequence_arrays,
    _zero_sketch_metrics,
)

def _rf_doc_features(doc: _CountDoc) -> np.ndarray:
    leaf = list(doc.leaf_features)
    if not leaf:
        raise ValueError("rf baseline requires at least one leaf per doc")
    feats = torch.stack(leaf, dim=0).to(dtype=torch.float32, device="cpu")
    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False)
    n_leaves = torch.tensor([float(feats.shape[0])], dtype=torch.float32)
    out = torch.cat([mean, std, n_leaves], dim=0).detach().cpu().numpy()
    return np.asarray(out, dtype=np.float64)


def _eval_rf_root_baseline(
    train_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    *,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> SketchMetrics:
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for include_rf_root_baseline. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    if not train_docs or not test_docs:
        return SketchMetrics(
            root_mae=float("nan"),
            root_median_abs_error=float("nan"),
            root_p95_abs_error=float("nan"),
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=float("nan"),
            leaf_violation_rate=float("nan"),
            c2_idempotence_mae=float("nan"),
            c2_r1_mae=float("nan"),
            c2_r2_mae=float("nan"),
            c2_r4_mae=float("nan"),
            resummary_root_drift_r1=float("nan"),
            resummary_root_drift_r2=float("nan"),
            resummary_root_drift_r4=float("nan"),
            merge_mae=float("nan"),
            merge_violation_rate=float("nan"),
            n_docs=int(len(test_docs)),
        )

    ne = int(n_estimators)
    if ne <= 0:
        raise ValueError("rf_n_estimators must be positive")
    md = int(max_depth)
    if md <= 0:
        raise ValueError("rf_max_depth must be positive")
    msl = int(min_samples_leaf)
    if msl <= 0:
        raise ValueError("rf_min_samples_leaf must be positive")

    X_train = np.stack([_rf_doc_features(d) for d in train_docs], axis=0).astype(
        np.float32, copy=False
    )
    y_train = np.asarray([float(d.root_count) for d in train_docs], dtype=np.float64)
    X_test = np.stack([_rf_doc_features(d) for d in test_docs], axis=0).astype(
        np.float32, copy=False
    )
    y_test = np.asarray([float(d.root_count) for d in test_docs], dtype=np.float64)

    model = RandomForestRegressor(
        n_estimators=int(ne),
        max_depth=int(md),
        min_samples_leaf=int(msl),
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_test), dtype=np.float64)
    abs_err = np.abs(pred - y_test)

    return SketchMetrics(
        root_mae=float(np.mean(abs_err)),
        root_median_abs_error=float(np.median(abs_err)),
        root_p95_abs_error=float(np.percentile(abs_err, 95.0)),
        schedule_spread_mean=0.0,
        schedule_spread_p95=0.0,
        leaf_mae=float("nan"),
        leaf_violation_rate=float("nan"),
        c2_idempotence_mae=float("nan"),
        c2_r1_mae=float("nan"),
        c2_r2_mae=float("nan"),
        c2_r4_mae=float("nan"),
        resummary_root_drift_r1=float("nan"),
        resummary_root_drift_r2=float("nan"),
        resummary_root_drift_r4=float("nan"),
        merge_mae=float("nan"),
        merge_violation_rate=float("nan"),
        n_docs=int(len(test_docs)),
    )


def _fit_doc_level_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    target_scale: float,
    device: torch.device,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    train_supervision: Optional[SupervisionDataset] = None,
    val_supervision: Optional[SupervisionDataset] = None,
    test_supervision: Optional[SupervisionDataset] = None,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    train_supervision = train_supervision or _doc_level_supervision_dataset(
        train_docs,
        split="train",
        target_scale=float(target_scale),
    )
    val_supervision = val_supervision or _doc_level_supervision_dataset(
        val_docs,
        split="val",
        target_scale=float(target_scale),
    )
    test_supervision = test_supervision or _doc_level_supervision_dataset(
        test_docs,
        split="test",
        target_scale=float(target_scale),
    )
    hidden_dims = tuple() if str(config.model_family) == "additive" else (int(config.hidden_dim),)
    model, fit_result = fit_dense_scalar_regressor(
        train_supervision,
        val_supervision=val_supervision if len(val_supervision) > 0 else None,
        config=DenseScalarTrainingConfig(
            model=DenseScalarModelConfig(hidden_dims=hidden_dims),
            train=TrainConfig(batch_size=int(config.batch_size), epochs=int(config.n_epochs)),
            optimizer=OptimizerConfig(
                learning_rate=float(config.lr),
                weight_decay=float(config.weight_decay),
                grad_clip_norm=float(config.grad_clip_norm),
            ),
            objective=DenseScalarObjectiveConfig(loss_name="mse"),
            runtime=RuntimeConfig(
                device=str(device),
                bf16=False,
                gradient_checkpointing=False,
            ),
            run=RunConfig(seed=int(seeds["effective_model_seed"]) + 40_003),
        ),
    )
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(fit_result.train_loss_final),
        train_loss_curve=tuple(float(x) for x in fit_result.train_loss_curve),
        epochs_completed=int(fit_result.epochs_completed),
        selection_metric_curve=tuple(float(x) for x in fit_result.selection_metric_curve),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
    )
    train_metrics = _eval_doc_level_dense_predictions(
        predict_dense_scalar_regressor(
            model,
            supervision=train_supervision,
            device=str(device),
        ),
        train_docs,
        tau=float(config.violation_tau),
    )
    val_metrics = _eval_doc_level_dense_predictions(
        predict_dense_scalar_regressor(
            model,
            supervision=val_supervision,
            device=str(device),
        ),
        val_docs,
        tau=float(config.violation_tau),
    )
    test_metrics = _eval_doc_level_dense_predictions(
        predict_dense_scalar_regressor(
            model,
            supervision=test_supervision,
            device=str(device),
        ),
        test_docs,
        tau=float(config.violation_tau),
    )
    return train_metrics, val_metrics, test_metrics, fit_diag


def _fit_doc_level_ridge_baseline(
    *,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    train_supervision: SupervisionDataset,
    ridge_alpha: float,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    model, fit_result = fit_dense_scalar_ridge_regressor(
        train_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(ridge_alpha))
        ),
    )

    def _metrics_for(docs: Sequence[_CountDoc]) -> SketchMetrics:
        if not docs:
            return _zero_sketch_metrics(n_docs=0)
        preds = predict_dense_scalar_ridge_regressor(
            model,
            _doc_level_feature_matrix(docs),
        )
        return _eval_doc_level_dense_predictions(preds, docs, tau=float(tau))

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float("nan"),
        train_loss_curve=tuple(),
        epochs_completed=0,
        selection_metric_curve=tuple(),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
    )
    return (
        _metrics_for(train_docs),
        _metrics_for(val_docs),
        _metrics_for(test_docs),
        fit_diag,
    )


def _fit_doc_token_ngram_ridge_baseline(
    *,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
    vocab_size: int,
    orders: Sequence[int],
    ridge_alpha: float,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    normalized_orders = _normalized_ngram_orders(orders)
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )
    train_x = _doc_token_ngram_feature_matrix(
        train_docs,
        vocab_size=int(vocab_size),
        orders=normalized_orders,
    )
    train_y = _doc_root_targets(train_docs)
    train_supervision = _dense_doc_matrix_supervision_dataset(
        train_x,
        train_y,
        split="train",
        target_scale=float(max(1.0, np.max(train_y))),
        input_view=f"full_document_token_{'_'.join(_ngram_order_label(order) for order in normalized_orders)}_counts",
        metadata={
            "ngram_orders": [int(order) for order in normalized_orders],
        },
    )
    model, fit_result = fit_dense_scalar_ridge_regressor(
        train_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(ridge_alpha))
        ),
    )

    def _metrics_for(docs: Sequence[ChangepointMarkovDoc]) -> SketchMetrics:
        if not docs:
            return _zero_sketch_metrics(n_docs=0)
        x = _doc_token_ngram_feature_matrix(
            docs,
            vocab_size=int(vocab_size),
            orders=normalized_orders,
        )
        preds = predict_dense_scalar_ridge_regressor(model, x)
        return _eval_root_predictions(
            preds,
            _doc_root_targets(docs).tolist(),
            tau=float(tau),
        )

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float("nan"),
        train_loss_curve=tuple(),
        epochs_completed=0,
        selection_metric_curve=tuple(),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
    )
    return (
        _metrics_for(train_docs),
        _metrics_for(val_docs),
        _metrics_for(test_docs),
        fit_diag,
    )


def _fit_doc_sequence_ctreepo_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    pad_id = int(config.vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)
    target_max = float(
        max(
            1.0,
            float(np.max(train_y)) if train_y.size > 0 else 0.0,
            float(np.max(val_y)) if val_y.size > 0 else 0.0,
            float(np.max(test_y)) if test_y.size > 0 else 0.0,
        )
    )
    class_values = sorted(
        {int(round(float(value))) for value in train_y.tolist()}
        | {int(round(float(value))) for value in val_y.tolist()}
    )
    if not class_values:
        class_values = [0]
    class_index = {int(value): idx for idx, value in enumerate(class_values)}

    model = FullSequenceCTreePOOperator(
        vocab_size=int(config.vocab_size),
        token_embedding_dim=max(64, int(config.state_dim)),
        sketch_dim=max(64, int(config.state_dim)),
        hidden_dim=max(256, int(config.hidden_dim)),
        target_max=float(target_max),
        n_count_classes=len(class_values),
    ).to(device=device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    rng = np.random.default_rng(int(seeds["effective_model_seed"]) + 50_003)
    batch_size = int(max(1, min(int(config.batch_size), int(train_tokens.shape[0]))))
    train_tokens_t = torch.tensor(train_tokens, dtype=torch.long, device=device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_tokens_t = torch.tensor(val_tokens, dtype=torch.long, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.float32, device=device)
    test_tokens_t = torch.tensor(test_tokens, dtype=torch.long, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.float32, device=device)
    train_target_norm = torch.tensor(
        [normalize_target(float(value), 0.0, float(target_max)) for value in train_y.tolist()],
        dtype=torch.float32,
        device=device,
    )
    train_target_class = torch.tensor(
        [int(class_index[int(round(float(value)))]) for value in train_y.tolist()],
        dtype=torch.long,
        device=device,
    )
    objective_name = str(config.doc_sequence_objective)

    best_state = clone_module_state(model)
    best_selection = TrainingSelectionMetadata(
        mode="best_val_root_mae_full_sequence_operator" if len(val_docs) > 0 else "final_epoch_no_validation",
        split="val" if len(val_docs) > 0 else "config",
        metric_name="val_root_mae_full_sequence_operator" if len(val_docs) > 0 else "train_root_mae_full_sequence_operator",
        metric_value=float("inf"),
        best_epoch=0,
    )
    loss_curve: List[float] = []
    selection_curve: List[float] = []

    class_values_arr = np.asarray([float(value) for value in class_values], dtype=np.float64)

    def _predict_counts(tokens_t: torch.Tensor, mask_t: torch.Tensor) -> np.ndarray:
        if int(tokens_t.shape[0]) <= 0:
            return np.zeros((0,), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            logits = model.predict_count_logits(tokens_t, token_mask=mask_t)
            pred_idx = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
        return class_values_arr[pred_idx]

    for epoch_idx in range(int(config.n_epochs)):
        model.train()
        perm = rng.permutation(int(train_tokens.shape[0]))
        batch_losses: List[float] = []
        for start in range(0, int(train_tokens.shape[0]), batch_size):
            batch_idx_np = perm[start : start + batch_size]
            batch_idx = torch.tensor(batch_idx_np, dtype=torch.long, device=device)
            batch_tokens = train_tokens_t.index_select(0, batch_idx)
            batch_mask = train_mask_t.index_select(0, batch_idx)
            batch_target_norm = train_target_norm.index_select(0, batch_idx)
            batch_target_class = train_target_class.index_select(0, batch_idx)
            opt.zero_grad(set_to_none=True)
            pred_logits = model.predict_count_logits(batch_tokens, token_mask=batch_mask)
            loss = F.cross_entropy(pred_logits, batch_target_class)
            if objective_name == "count_ce_plus_scalar_mse":
                pred_norm = model.predict_normalized(batch_tokens, token_mask=batch_mask)
                loss = loss + 0.25 * F.mse_loss(pred_norm, batch_target_norm, reduction="mean")
            loss.backward()
            if float(config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_train_loss = float(np.mean(np.asarray(batch_losses, dtype=np.float64)))
        loss_curve.append(epoch_train_loss)

        if len(val_docs) > 0:
            val_pred = _predict_counts(val_tokens_t, val_mask_t)
            selection_value = float(np.mean(np.abs(val_pred - val_y.astype(np.float64, copy=False))))
            selection_curve.append(selection_value)
            if improved_metric(selection_value, best_selection.metric_value):
                best_selection = TrainingSelectionMetadata(
                    mode="best_val_root_mae_full_sequence_operator",
                    split="val",
                    metric_name="val_root_mae_full_sequence_operator",
                    metric_value=float(selection_value),
                    best_epoch=int(epoch_idx),
                )
                best_state = clone_module_state(model)
        else:
            train_pred_epoch = _predict_counts(train_tokens_t, train_mask_t)
            selection_value = float(np.mean(np.abs(train_pred_epoch - train_y.astype(np.float64, copy=False))))
            selection_curve.append(selection_value)

    if len(val_docs) > 0:
        restore_module_state(model, best_state)
    else:
        final_train_pred = _predict_counts(train_tokens_t, train_mask_t)
        final_train_mae = float(np.mean(np.abs(final_train_pred - train_y.astype(np.float64, copy=False))))
        best_selection = TrainingSelectionMetadata(
            mode="final_epoch_no_validation",
            split="config",
            metric_name="train_root_mae_full_sequence_operator",
            metric_value=float(final_train_mae),
            best_epoch=max(0, int(len(loss_curve) - 1)),
        )

    train_pred = _predict_counts(train_tokens_t, train_mask_t)
    val_pred = _predict_counts(val_tokens_t, val_mask_t)
    test_pred = _predict_counts(test_tokens_t, test_mask_t)
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(loss_curve[-1]) if loss_curve else float("nan"),
        train_loss_curve=tuple(float(x) for x in loss_curve),
        epochs_completed=int(len(loss_curve)),
        selection_metric_curve=tuple(float(x) for x in selection_curve),
        selection_mode=str(best_selection.mode),
        selection_split=str(best_selection.split),
        selection_metric_name=str(best_selection.metric_name),
        selection_metric_value=float(best_selection.metric_value),
        best_epoch=int(best_selection.best_epoch),
        train_exact_match_rate=float(_exact_match_rate(train_pred, train_y.tolist())),
        val_exact_match_rate=float(_exact_match_rate(val_pred, val_y.tolist())),
        test_exact_match_rate=float(_exact_match_rate(test_pred, test_y.tolist())),
    )
    return (
        _eval_root_predictions(train_pred, train_y.tolist(), tau=float(config.violation_tau)),
        _eval_root_predictions(val_pred, val_y.tolist(), tau=float(config.violation_tau)),
        _eval_root_predictions(test_pred, test_y.tolist(), tau=float(config.violation_tau)),
        fit_diag,
    )


def _fit_doc_sequence_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import _fit_fno_baseline

    return _fit_fno_baseline(
        config=config,
        seeds=seeds,
        device=device,
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
    )


def _fit_doc_transformer_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    pad_id = int(config.vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)
    target_max = float(
        max(
            1.0,
            float(np.max(train_y)) if train_y.size > 0 else 0.0,
            float(np.max(val_y)) if val_y.size > 0 else 0.0,
            float(np.max(test_y)) if test_y.size > 0 else 0.0,
        )
    )
    class_values = sorted(
        {int(round(float(value))) for value in train_y.tolist()}
        | {int(round(float(value))) for value in val_y.tolist()}
    )
    if not class_values:
        class_values = [0]
    class_index = {int(value): idx for idx, value in enumerate(class_values)}

    model_dim = int(max(64, min(512, max(int(config.state_dim), int(config.hidden_dim) // 2))))
    n_heads = 8 if model_dim % 8 == 0 else 4
    while model_dim % n_heads != 0:
        model_dim += 1
    model = FullSequenceBoundaryTransformer(
        vocab_size=int(config.vocab_size),
        max_positions=int(max(8, train_tokens.shape[1], val_tokens.shape[1], test_tokens.shape[1])),
        model_dim=model_dim,
        hidden_dim=max(256, int(config.hidden_dim)),
        n_layers=(
            int(config.doc_transformer_layers)
            if int(config.doc_transformer_layers) > 0
            else (4 if model_dim >= 128 else 3)
        ),
        n_heads=n_heads,
        n_count_classes=len(class_values),
    ).to(device=device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    rng = np.random.default_rng(int(seeds["effective_model_seed"]) + 70_003)
    batch_size = int(max(1, min(int(config.batch_size), int(train_tokens.shape[0]))))
    train_tokens_t = torch.tensor(train_tokens, dtype=torch.long, device=device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_tokens_t = torch.tensor(val_tokens, dtype=torch.long, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.float32, device=device)
    test_tokens_t = torch.tensor(test_tokens, dtype=torch.long, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.float32, device=device)
    train_target_count = torch.tensor(train_y, dtype=torch.float32, device=device)
    train_target_class = torch.tensor(
        [int(class_index[int(round(float(value)))]) for value in train_y.tolist()],
        dtype=torch.long,
        device=device,
    )
    head_family = str(config.doc_transformer_head_family)
    class_values_arr = np.asarray([float(value) for value in class_values], dtype=np.float64)

    best_state = clone_module_state(model)
    best_selection = TrainingSelectionMetadata(
        mode=(
            "best_val_root_mae_full_sequence_boundary_transformer"
            if len(val_docs) > 0
            else "final_epoch_no_validation"
        ),
        split="val" if len(val_docs) > 0 else "config",
        metric_name=(
            "val_root_mae_full_sequence_boundary_transformer"
            if len(val_docs) > 0
            else "train_root_mae_full_sequence_boundary_transformer"
        ),
        metric_value=float("inf"),
        best_epoch=0,
    )
    loss_curve: List[float] = []
    selection_curve: List[float] = []

    def _predict_counts(tokens_t: torch.Tensor, mask_t: torch.Tensor) -> np.ndarray:
        if int(tokens_t.shape[0]) <= 0:
            return np.zeros((0,), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            if head_family == "pooled_count_classifier":
                logits = model.predict_count_logits(tokens_t, token_mask=mask_t)
                pred_idx = (
                    torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
                )
                return class_values_arr[pred_idx]
            pred = (
                model(tokens_t, token_mask=mask_t).detach().cpu().numpy().astype(np.float64, copy=False)
            )
        return pred

    for epoch_idx in range(int(config.n_epochs)):
        model.train()
        perm = rng.permutation(int(train_tokens.shape[0]))
        batch_losses: List[float] = []
        for start in range(0, int(train_tokens.shape[0]), batch_size):
            batch_idx_np = perm[start : start + batch_size]
            batch_idx = torch.tensor(batch_idx_np, dtype=torch.long, device=device)
            batch_tokens = train_tokens_t.index_select(0, batch_idx)
            batch_mask = train_mask_t.index_select(0, batch_idx)
            batch_target_count = train_target_count.index_select(0, batch_idx)
            batch_target_class = train_target_class.index_select(0, batch_idx)
            opt.zero_grad(set_to_none=True)
            pred_count_logits = model.predict_count_logits(batch_tokens, token_mask=batch_mask)
            if head_family == "pooled_count_classifier":
                loss = F.cross_entropy(pred_count_logits, batch_target_class)
            else:
                pred_count = model(batch_tokens, token_mask=batch_mask)
                loss = F.mse_loss(pred_count, batch_target_count, reduction="mean")
                loss = loss + 0.35 * F.cross_entropy(pred_count_logits, batch_target_class)
            loss.backward()
            if float(config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_train_loss = float(np.mean(np.asarray(batch_losses, dtype=np.float64)))
        loss_curve.append(epoch_train_loss)

        if len(val_docs) > 0:
            val_pred = _predict_counts(val_tokens_t, val_mask_t)
            selection_value = float(np.mean(np.abs(val_pred - val_y.astype(np.float64, copy=False))))
            selection_curve.append(selection_value)
            if improved_metric(selection_value, best_selection.metric_value):
                best_selection = TrainingSelectionMetadata(
                    mode="best_val_root_mae_full_sequence_boundary_transformer",
                    split="val",
                    metric_name="val_root_mae_full_sequence_boundary_transformer",
                    metric_value=float(selection_value),
                    best_epoch=int(epoch_idx),
                )
                best_state = clone_module_state(model)
        else:
            train_pred_epoch = _predict_counts(train_tokens_t, train_mask_t)
            selection_value = float(np.mean(np.abs(train_pred_epoch - train_y.astype(np.float64, copy=False))))
            selection_curve.append(selection_value)

    if len(val_docs) > 0:
        restore_module_state(model, best_state)
    else:
        final_train_pred = _predict_counts(train_tokens_t, train_mask_t)
        final_train_mae = float(np.mean(np.abs(final_train_pred - train_y.astype(np.float64, copy=False))))
        best_selection = TrainingSelectionMetadata(
            mode="final_epoch_no_validation",
            split="config",
            metric_name="train_root_mae_full_sequence_boundary_transformer",
            metric_value=float(final_train_mae),
            best_epoch=max(0, int(len(loss_curve) - 1)),
        )

    train_pred = _predict_counts(train_tokens_t, train_mask_t)
    val_pred = _predict_counts(val_tokens_t, val_mask_t)
    test_pred = _predict_counts(test_tokens_t, test_mask_t)
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(loss_curve[-1]) if loss_curve else float("nan"),
        train_loss_curve=tuple(float(x) for x in loss_curve),
        epochs_completed=int(len(loss_curve)),
        selection_metric_curve=tuple(float(x) for x in selection_curve),
        selection_mode=str(best_selection.mode),
        selection_split=str(best_selection.split),
        selection_metric_name=str(best_selection.metric_name),
        selection_metric_value=float(best_selection.metric_value),
        best_epoch=int(best_selection.best_epoch),
        train_exact_match_rate=float(_exact_match_rate(train_pred, train_y.tolist())),
        val_exact_match_rate=float(_exact_match_rate(val_pred, val_y.tolist())),
        test_exact_match_rate=float(_exact_match_rate(test_pred, test_y.tolist())),
    )
    return (
        _eval_root_predictions(train_pred, train_y.tolist(), tau=float(config.violation_tau)),
        _eval_root_predictions(val_pred, val_y.tolist(), tau=float(config.violation_tau)),
        _eval_root_predictions(test_pred, test_y.tolist(), tau=float(config.violation_tau)),
        fit_diag,
    )


def _fit_sampled_leaf_pool_ridge_baseline(
    *,
    train_docs: Sequence[_SampledLeafPoolDoc],
    val_docs: Sequence[_SampledLeafPoolDoc],
    test_docs: Sequence[_SampledLeafPoolDoc],
    train_supervision: SupervisionDataset,
    ridge_alpha: float,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    model, fit_result = fit_dense_scalar_ridge_regressor(
        train_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(ridge_alpha))
        ),
    )

    def _metrics_for(docs: Sequence[_SampledLeafPoolDoc]) -> SketchMetrics:
        if not docs:
            return _zero_sketch_metrics(n_docs=0)
        preds = predict_dense_scalar_ridge_regressor(
            model,
            _sampled_leaf_pool_feature_matrix(docs),
        )
        return _eval_root_only_predictions(
            preds,
            [float(doc.root_count) for doc in docs],
            n_docs=int(len(docs)),
            tau=float(tau),
        )

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float("nan"),
        train_loss_curve=tuple(),
        epochs_completed=0,
        selection_metric_curve=tuple(),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
    )
    return (
        _metrics_for(train_docs),
        _metrics_for(val_docs),
        _metrics_for(test_docs),
        fit_diag,
    )


def _fit_leaf_ridge_tree_baseline(
    *,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    train_supervision: SupervisionDataset,
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    ridge_alpha: float,
    device: torch.device,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    feature_dim = int(train_docs[0].leaf_features[0].numel())
    model = AdditiveCountSketch(
        feature_dim=int(feature_dim),
        hidden_dim=1,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        use_endpoints=bool(use_endpoints),
    ).to(device=device)
    ridge_model, fit_result = fit_dense_scalar_ridge_regressor(
        train_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(ridge_alpha))
        ),
    )
    with torch.no_grad():
        model.encoder.weight.copy_(
            torch.tensor(
                np.asarray(ridge_model.weights, dtype=np.float64).reshape(1, -1),
                device=device,
                dtype=torch.float32,
            )
        )
        model.encoder.bias.copy_(
            torch.tensor([float(ridge_model.bias)], device=device, dtype=torch.float32)
        )

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float("nan"),
        train_loss_curve=tuple(),
        epochs_completed=0,
        selection_metric_curve=tuple(),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
    )
    return (
        _eval_learned_model(model, train_docs, device=device, tau=float(tau)),
        _eval_learned_model(model, val_docs, device=device, tau=float(tau)),
        _eval_learned_model(model, test_docs, device=device, tau=float(tau)),
        fit_diag,
    )


def _balanced_additive_merge_counts(
    leaf_counts: Sequence[float],
    *,
    leaf_features: Sequence[torch.Tensor],
    n_regimes: int,
    use_endpoints: bool,
) -> tuple[float, List[float]]:
    if len(leaf_counts) != len(leaf_features):
        raise ValueError("leaf_counts and leaf_features must align")
    if not leaf_counts:
        raise ValueError("need at least one leaf count")

    if bool(use_endpoints):
        states: List[tuple[float, int, int]] = []
        n = int(n_regimes)
        for count, feature in zip(leaf_counts, leaf_features):
            first_id = int(torch.argmax(feature[:n]).item())
            last_id = int(torch.argmax(feature[n : 2 * n]).item())
            states.append((float(count), int(first_id), int(last_id)))
    else:
        states = [(float(count), 0, 0) for count in leaf_counts]

    merge_counts: List[float] = []
    cur = list(states)
    while len(cur) > 1:
        nxt: List[tuple[float, int, int]] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
                continue
            left_count, left_first, left_last = cur[i]
            right_count, right_first, right_last = cur[i + 1]
            join = 0.0
            if bool(use_endpoints) and int(left_last) != int(right_first):
                join = 1.0
            merged_count = float(left_count) + float(right_count) + float(join)
            merge_counts.append(float(merged_count))
            nxt.append((float(merged_count), int(left_first), int(right_last)))
            i += 2
        cur = nxt
    return float(cur[0][0]), merge_counts


def _eval_leaf_local_additive_predictor(
    *,
    docs: Sequence[_CountDoc],
    predict_leaf_norm: Callable[[np.ndarray], float],
    local_feature_builder: Callable[[torch.Tensor], np.ndarray],
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    tau: float,
) -> SketchMetrics:
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    root_abs: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

    for doc in docs:
        pred_leaf_counts: List[float] = []
        for feature, truth in zip(doc.leaf_features, doc.leaf_counts):
            pred_norm = float(predict_leaf_norm(local_feature_builder(feature)))
            pred_count = float(pred_norm) * float(target_scale)
            pred_leaf_counts.append(float(pred_count))
            leaf_abs.append(abs(float(pred_count) - float(truth)))

        pred_root, pred_merges = _balanced_additive_merge_counts(
            pred_leaf_counts,
            leaf_features=doc.leaf_features,
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
        )
        root_abs.append(abs(float(pred_root) - float(doc.root_count)))
        for pred_merge, truth in zip(pred_merges, doc.merge_counts_balanced):
            merge_abs.append(abs(float(pred_merge) - float(truth)))

    root_arr = np.asarray(root_abs, dtype=np.float64)
    leaf_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_arr = np.asarray(merge_abs, dtype=np.float64)
    tau_v = float(tau)
    return SketchMetrics(
        root_mae=float(np.mean(root_arr)),
        root_median_abs_error=float(np.median(root_arr)),
        root_p95_abs_error=float(np.percentile(root_arr, 95.0)),
        schedule_spread_mean=0.0,
        schedule_spread_p95=0.0,
        leaf_mae=float(np.mean(leaf_arr)),
        leaf_violation_rate=float(np.mean((leaf_arr > tau_v).astype(np.float64))),
        c2_idempotence_mae=0.0,
        c2_r1_mae=0.0,
        c2_r2_mae=0.0,
        c2_r4_mae=0.0,
        resummary_root_drift_r1=0.0,
        resummary_root_drift_r2=0.0,
        resummary_root_drift_r4=0.0,
        merge_mae=float(np.mean(merge_arr)) if merge_arr.size else 0.0,
        merge_violation_rate=(
            float(np.mean((merge_arr > tau_v).astype(np.float64))) if merge_arr.size else 0.0
        ),
        n_docs=int(len(docs)),
    )


def _fit_leaf_knn_tree_baseline(
    *,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    train_supervision: SupervisionDataset,
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    n_neighbors: int,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )
    try:
        from sklearn.neighbors import KNeighborsRegressor  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for include_leaf_knn_tree_baseline. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    rows = dense_scalar_rows(train_supervision)
    if not rows:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )
    x_train, y_train, _w = dense_scalar_rows_to_numpy(rows)
    k = int(max(1, min(int(n_neighbors), int(x_train.shape[0]))))
    model = KNeighborsRegressor(
        n_neighbors=int(k),
        weights="distance",
        metric="euclidean",
    )
    model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32))
    train_pred = np.asarray(model.predict(np.asarray(x_train, dtype=np.float32)), dtype=np.float64)
    train_mse = float(np.mean((train_pred - np.asarray(y_train, dtype=np.float64)) ** 2))

    def _predict_leaf_norm(features: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(np.asarray(model.predict(x), dtype=np.float64).reshape(-1)[0])

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(train_mse),
        train_loss_curve=(float(train_mse),),
        epochs_completed=0,
        selection_metric_curve=(float(train_mse),),
        selection_mode="knn_fit_no_validation",
        selection_split="config",
        selection_metric_name="train_mse",
        selection_metric_value=float(train_mse),
        best_epoch=0,
    )
    return (
        _eval_leaf_local_additive_predictor(
            docs=train_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=lambda feature: _leaf_core_feature_vector(
                feature,
                n_regimes=int(n_regimes),
                use_endpoints=bool(use_endpoints),
            ),
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=val_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=lambda feature: _leaf_core_feature_vector(
                feature,
                n_regimes=int(n_regimes),
                use_endpoints=bool(use_endpoints),
            ),
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=test_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=lambda feature: _leaf_core_feature_vector(
                feature,
                n_regimes=int(n_regimes),
                use_endpoints=bool(use_endpoints),
            ),
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        fit_diag,
    )


def _fit_leaf_endpoint_table_tree_baseline(
    *,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    train_supervision: SupervisionDataset,
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    rows = dense_scalar_rows(train_supervision)
    if not rows:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    x_train, y_train, _w = dense_scalar_rows_to_numpy(rows)
    y_train_arr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    global_mean = float(np.mean(y_train_arr)) if y_train_arr.size else 0.0
    sum_by_key: Dict[Tuple[int, int, int], float] = {}
    count_by_key: Dict[Tuple[int, int, int], int] = {}
    for x_row, y_value in zip(np.asarray(x_train, dtype=np.float64), y_train_arr):
        key = _leaf_endpoint_table_key(x_row)
        sum_by_key[key] = float(sum_by_key.get(key, 0.0) + float(y_value))
        count_by_key[key] = int(count_by_key.get(key, 0) + 1)
    mean_by_key = {
        key: float(sum_by_key[key] / float(max(1, count_by_key[key])))
        for key in sum_by_key
    }

    def _predict_leaf_norm(local_features: np.ndarray) -> float:
        return float(mean_by_key.get(_leaf_endpoint_table_key(local_features), global_mean))

    train_pred = np.asarray(
        [_predict_leaf_norm(np.asarray(x_row, dtype=np.float64)) for x_row in x_train],
        dtype=np.float64,
    )
    train_mse = float(np.mean((train_pred - y_train_arr) ** 2)) if y_train_arr.size else float("nan")
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(train_mse),
        train_loss_curve=(float(train_mse),),
        epochs_completed=0,
        selection_metric_curve=(float(train_mse),),
        selection_mode="group_mean_lookup_no_validation",
        selection_split="config",
        selection_metric_name="train_mse",
        selection_metric_value=float(train_mse),
        best_epoch=0,
    )
    feature_builder = lambda feature: _leaf_endpoint_table_feature_vector(
        feature,
        n_regimes=int(n_regimes),
        use_endpoints=bool(use_endpoints),
    )
    return (
        _eval_leaf_local_additive_predictor(
            docs=train_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=feature_builder,
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=val_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=feature_builder,
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=test_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=feature_builder,
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        fit_diag,
    )


def _fit_leaf_dt_tree_baseline(
    *,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    train_supervision: SupervisionDataset,
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    seed: int,
    max_depth: int,
    min_samples_leaf: int,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    try:
        from sklearn.tree import DecisionTreeRegressor  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for include_leaf_dt_tree_baseline. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    rows = dense_scalar_rows(train_supervision)
    if not rows:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    x_train, y_train, _w = dense_scalar_rows_to_numpy(rows)
    model = DecisionTreeRegressor(
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
    )
    x_train_f32 = np.asarray(x_train, dtype=np.float32)
    y_train_f64 = np.asarray(y_train, dtype=np.float64)
    model.fit(x_train_f32, y_train_f64)
    train_pred = np.asarray(model.predict(x_train_f32), dtype=np.float64)
    train_mse = float(np.mean((train_pred - y_train_f64) ** 2))

    def _predict_leaf_norm(features: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(np.asarray(model.predict(x), dtype=np.float64).reshape(-1)[0])

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(train_mse),
        train_loss_curve=(float(train_mse),),
        epochs_completed=0,
        selection_metric_curve=(float(train_mse),),
        selection_mode="dt_fit_no_validation",
        selection_split="config",
        selection_metric_name="train_mse",
        selection_metric_value=float(train_mse),
        best_epoch=0,
    )
    return (
        _eval_leaf_local_additive_predictor(
            docs=train_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=lambda feature: _leaf_core_feature_vector(
                feature,
                n_regimes=int(n_regimes),
                use_endpoints=bool(use_endpoints),
            ),
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=val_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=lambda feature: _leaf_core_feature_vector(
                feature,
                n_regimes=int(n_regimes),
                use_endpoints=bool(use_endpoints),
            ),
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=test_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=lambda feature: _leaf_core_feature_vector(
                feature,
                n_regimes=int(n_regimes),
                use_endpoints=bool(use_endpoints),
            ),
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        fit_diag,
    )


def _fit_leaf_rf_tree_baseline(
    *,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    train_supervision: SupervisionDataset,
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for include_leaf_rf_tree_baseline. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    rows = dense_scalar_rows(train_supervision)
    if not rows:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return (
            zero_train,
            zero_val,
            zero_test,
            TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
        )

    x_train, y_train, _w = dense_scalar_rows_to_numpy(rows)
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=1,
    )
    x_train_f32 = np.asarray(x_train, dtype=np.float32)
    y_train_f64 = np.asarray(y_train, dtype=np.float64)
    model.fit(x_train_f32, y_train_f64)
    train_pred = np.asarray(model.predict(x_train_f32), dtype=np.float64)
    train_mse = float(np.mean((train_pred - y_train_f64) ** 2))

    def _predict_leaf_norm(features: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(np.asarray(model.predict(x), dtype=np.float64).reshape(-1)[0])

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(train_mse),
        train_loss_curve=(float(train_mse),),
        epochs_completed=0,
        selection_metric_curve=(float(train_mse),),
        selection_mode="rf_fit_no_validation",
        selection_split="config",
        selection_metric_name="train_mse",
        selection_metric_value=float(train_mse),
        best_epoch=0,
    )
    feature_builder = lambda feature: _leaf_core_feature_vector(
        feature,
        n_regimes=int(n_regimes),
        use_endpoints=bool(use_endpoints),
    )
    return (
        _eval_leaf_local_additive_predictor(
            docs=train_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=feature_builder,
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=val_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=feature_builder,
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        _eval_leaf_local_additive_predictor(
            docs=test_docs,
            predict_leaf_norm=_predict_leaf_norm,
            local_feature_builder=feature_builder,
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            use_endpoints=bool(use_endpoints),
            tau=float(tau),
        ),
        fit_diag,
    )


def _fit_sampled_leaf_pool_rf_baseline(
    train_docs: Sequence[_SampledLeafPoolDoc],
    val_docs: Sequence[_SampledLeafPoolDoc],
    test_docs: Sequence[_SampledLeafPoolDoc],
    *,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    tau: float,
) -> tuple[SketchMetrics, SketchMetrics, SketchMetrics]:
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for include_sampled_leaf_pool_rf_baseline. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    if not train_docs:
        zero_train = _zero_sketch_metrics(n_docs=int(len(train_docs)))
        zero_val = _zero_sketch_metrics(n_docs=int(len(val_docs)))
        zero_test = _zero_sketch_metrics(n_docs=int(len(test_docs)))
        return zero_train, zero_val, zero_test

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(
        _sampled_leaf_pool_feature_matrix(train_docs).astype(np.float32, copy=False),
        np.asarray([float(doc.root_count) for doc in train_docs], dtype=np.float64),
    )

    def _metrics_for(docs: Sequence[_SampledLeafPoolDoc]) -> SketchMetrics:
        if not docs:
            return _zero_sketch_metrics(n_docs=0)
        pred = np.asarray(
            model.predict(_sampled_leaf_pool_feature_matrix(docs).astype(np.float32, copy=False)),
            dtype=np.float64,
        )
        return _eval_root_only_predictions(
            pred,
            [float(doc.root_count) for doc in docs],
            n_docs=int(len(docs)),
            tau=float(tau),
        )

    return _metrics_for(train_docs), _metrics_for(val_docs), _metrics_for(test_docs)
