from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    OPSCountSummary,
    _doc_level_supervision_dataset,
    _eval_rf_root_baseline,
    _fit_doc_level_baseline,
    _fit_doc_level_ridge_baseline,
    _fit_doc_sequence_baseline,
    _fit_doc_token_ngram_ridge_baseline,
    _prepare_count_docs,
    _prepare_doc_level_count_docs,
    _resolve_runtime_seeds,
    _set_global_seed,
    build_markov_changepoint_ops_count_data_bundle,
)
from treepo._research.ctreepo.sim.core.markov_full_doc_provenance import (
    official_fno_doc_sequence_provenance,
)
from treepo._research.ctreepo.sim.suite.markov_observed_token_policy import (
    resolve_markov_observed_token_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[4]

CANONICAL_BUNDLES = {
    "demo_v1": REPO_ROOT
    / "outputs/markov_observed_token_suite_demo_v1/markov_data/observed_token_bundle.json",
    "recoverable": REPO_ROOT
    / "outputs/markov_observed_token_recoverable_v4/markov_data/observed_token_bundle.json",
}

REFERENCE_SUMMARIES = {
    "recoverable_official_fno_v1": REPO_ROOT
    / "outputs/markov_official_fno_recoverable_v1/operator_only.json",
}


@dataclass(frozen=True)
class MarkovFullDocAnchorStageSpec:
    name: str
    description: str
    observed_token_profile: str = ""
    bundle_path: str = ""
    summary_json: str = ""
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    reference_only: bool = False


def resolve_markov_full_doc_anchor_ladder(
    *,
    preset: str = "quick_buildout",
) -> tuple[MarkovFullDocAnchorStageSpec, ...]:
    key = str(preset or "quick_buildout").strip().lower() or "quick_buildout"
    if key == "quick_buildout":
        return (
            MarkovFullDocAnchorStageSpec(
                name="demo_v1_zero_reference",
                description=(
                    "Degenerate fixed-count demo endpoint where the standalone full-doc anchor "
                    "hits zero error."
                ),
                observed_token_profile="demo_v1",
                bundle_path=str(CANONICAL_BUNDLES["demo_v1"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_current_public",
                description=(
                    "Current public recoverable profile on the fixed disjoint-palette bundle."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_wider_operator_public",
                description=(
                    "Same recoverable bundle with a wider public operator and longer training."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 128,
                    "hidden_dim": 512,
                    "n_epochs": 96,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_official_fno_reproduction",
                description=(
                    "Fresh rerun of the larger official-FNO-style recoverable endpoint on the "
                    "same fixed bundle."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 256,
                    "hidden_dim": 1024,
                    "n_epochs": 128,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_official_fno_reference",
                description=(
                    "Historical public-API official FNO recoverable reference on the same fixed bundle."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                summary_json=str(REFERENCE_SUMMARIES["recoverable_official_fno_v1"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 256,
                    "hidden_dim": 1024,
                    "n_epochs": 128,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                    "train_docs": 1024,
                    "val_docs": 128,
                    "test_docs": 256,
                },
                reference_only=True,
            ),
        )
    if key == "recoverable_budget_ladder":
        return (
            MarkovFullDocAnchorStageSpec(
                name="recoverable_current_public",
                description="Recoverable default profile.",
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_wider_operator_public",
                description="Recoverable public run with wider state and hidden widths.",
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 128,
                    "hidden_dim": 512,
                    "n_epochs": 96,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_official_fno_reproduction",
                description=(
                    "Fresh rerun of the larger official-FNO-style recoverable endpoint."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 256,
                    "hidden_dim": 1024,
                    "n_epochs": 128,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_official_fno_reference",
                description="Historical public-API official FNO recoverable reference.",
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                summary_json=str(REFERENCE_SUMMARIES["recoverable_official_fno_v1"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 256,
                    "hidden_dim": 1024,
                    "n_epochs": 128,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                    "train_docs": 1024,
                    "val_docs": 128,
                    "test_docs": 256,
                },
                reference_only=True,
            ),
        )
    if key == "recoverable_reproduction_ladder":
        return (
            MarkovFullDocAnchorStageSpec(
                name="recoverable_current_public",
                description=(
                    "Current public recoverable profile on the fixed disjoint-palette bundle."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_medium_operator_public",
                description=(
                    "Recoverable bundle with doubled operator state and hidden widths."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 128,
                    "hidden_dim": 512,
                    "n_epochs": 72,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_wider_operator_public",
                description=(
                    "Recoverable bundle with doubled width and near-reference epoch budget."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 128,
                    "hidden_dim": 512,
                    "n_epochs": 96,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_official_fno_reproduction",
                description=(
                    "Fresh rerun of the larger official-FNO-style recoverable endpoint on the "
                    "same fixed bundle."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 256,
                    "hidden_dim": 1024,
                    "n_epochs": 128,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ),
            MarkovFullDocAnchorStageSpec(
                name="recoverable_official_fno_reference",
                description=(
                    "Historical public-API official FNO recoverable reference on the same fixed bundle."
                ),
                observed_token_profile="recoverable",
                bundle_path=str(CANONICAL_BUNDLES["recoverable"]),
                summary_json=str(REFERENCE_SUMMARIES["recoverable_official_fno_v1"]),
                config_overrides={
                    "model_family": "neural",
                    "feature_mode": "token_full",
                    "state_dim": 256,
                    "hidden_dim": 1024,
                    "n_epochs": 128,
                    "batch_size": 64,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                    "train_docs": 1024,
                    "val_docs": 128,
                    "test_docs": 256,
                },
                reference_only=True,
            ),
        )
    raise ValueError(f"unknown full-doc anchor ladder preset: {preset!r}")


def _summary_from_json_path(path: Path) -> OPSCountSummary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return OPSCountSummary(
        config=dict(payload.get("config") or {}),
        training_geometry=dict(payload.get("training_geometry") or {}),
        objective=dict(payload.get("objective") or {}),
        metrics=dict(payload.get("metrics") or {}),
        estimator_diagnostics=dict(payload.get("estimator_diagnostics") or {}),
        local_law_learnability=dict(payload.get("local_law_learnability") or {}),
        g_artifacts=dict(payload.get("g_artifacts") or {}),
    )


def _root_count_support(bundle: MarkovOPSDataBundle) -> Dict[str, Any]:
    def _split_values(docs: Sequence[Any]) -> List[int]:
        return sorted({int(len(doc.true_boundaries)) for doc in docs})

    train_values = _split_values(bundle.train_docs)
    val_values = _split_values(bundle.val_docs)
    test_values = _split_values(bundle.test_docs)
    return {
        "train_values": train_values,
        "val_values": val_values,
        "test_values": test_values,
        "all_values": sorted(set(train_values) | set(val_values) | set(test_values)),
    }


def _stage_config_from_profile(
    stage: MarkovFullDocAnchorStageSpec,
    *,
    output_dir: Path | None,
    use_cuda: bool,
    cuda_device: int | None,
    torch_threads: int,
) -> OPSCountConfig:
    if not str(stage.observed_token_profile).strip():
        raise ValueError("stage.observed_token_profile is required for runnable stages")
    policy = resolve_markov_observed_token_policy(
        profile_name=str(stage.observed_token_profile),
    )
    artifact_dir = ""
    if output_dir is not None:
        artifact_dir = str(output_dir / "artifacts" / stage.name)
    cfg = OPSCountConfig(
        n_regimes=int(policy.n_regimes),
        vocab_size=int(policy.vocab_size),
        generator_profile=str(policy.generator_profile),
        min_tokens=int(policy.min_tokens),
        max_tokens=int(policy.max_tokens),
        min_segments=int(policy.min_segments),
        max_segments=int(policy.max_segments),
        min_seg_len=int(policy.min_seg_len),
        max_seg_len=int(policy.max_seg_len),
        fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
        train_docs=int(policy.train_docs),
        val_docs=int(policy.val_docs),
        test_docs=int(policy.test_docs),
        model_family="neural",
        feature_mode="token_full",
        state_dim=int(policy.state_dim),
        hidden_dim=int(policy.hidden_dim),
        n_epochs=int(policy.n_epochs),
        batch_size=int(policy.batch_size),
        lr=float(policy.lr),
        weight_decay=float(policy.weight_decay),
        use_cuda=bool(use_cuda),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        torch_threads=int(torch_threads),
        include_doc_level_baseline=True,
        include_doc_level_ridge_baseline=True,
        include_doc_sequence_baseline=True,
        include_rf_root_baseline=True,
        doc_sequence_objective=str(policy.doc_sequence_objective),
        doc_transformer_head_family=str(policy.doc_transformer_head_family),
        doc_transformer_layers=int(policy.doc_transformer_layers),
        doc_level_ridge_alpha=float(policy.doc_level_ridge_alpha),
        doc_level_ridge_breakdown_orders=tuple(
            int(x) for x in policy.doc_level_ridge_breakdown_orders
        ),
        rf_n_estimators=int(policy.rf_n_estimators),
        rf_max_depth=int(policy.rf_max_depth),
        rf_min_samples_leaf=int(policy.rf_min_samples_leaf),
        leaf_knn_neighbors=int(policy.leaf_knn_neighbors),
        artifact_dir=artifact_dir,
        seed=int(policy.seed),
    )
    overrides = dict(stage.config_overrides or {})
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def _load_or_build_bundle(
    *,
    stage: MarkovFullDocAnchorStageSpec,
    cfg: OPSCountConfig,
    output_dir: Path | None,
) -> tuple[MarkovOPSDataBundle, str]:
    bundle_path = Path(str(stage.bundle_path)) if str(stage.bundle_path).strip() else None
    if bundle_path is not None and bundle_path.exists():
        return MarkovOPSDataBundle.load(bundle_path), str(bundle_path)
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    bundle_source = "generated"
    if output_dir is not None:
        out_path = output_dir / "bundles" / f"{stage.name}.json"
        bundle.save(out_path)
        bundle_source = str(out_path)
    return bundle, bundle_source


def _get_float(payload: Mapping[str, Any], key: str) -> float:
    return float(payload.get(key, float("nan")))


def _load_cached_stage_result(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cached stage result at {path} is not a JSON object")
    return dict(payload)


def _resolve_stage_device(config: OPSCountConfig) -> tuple[Dict[str, int], Any]:
    import torch

    seeds = _resolve_runtime_seeds(config)
    _set_global_seed(int(seeds["effective_model_seed"]))
    if int(config.torch_threads) > 0:
        torch.set_num_threads(int(config.torch_threads))
    if bool(config.use_cuda) and torch.cuda.is_available():
        if config.cuda_device is not None:
            idx = int(config.cuda_device)
            if idx < 0 or idx >= int(torch.cuda.device_count()):
                raise ValueError(f"cuda_device={idx} out of range")
            torch.cuda.set_device(idx)
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return seeds, device


def _run_full_doc_baselines_stage(
    *,
    stage: MarkovFullDocAnchorStageSpec,
    cfg: OPSCountConfig,
    bundle: MarkovOPSDataBundle,
    stage_summary_path: Path | None,
    bundle_source: str,
) -> Dict[str, Any]:
    docs_train = tuple(bundle.train_docs[: int(cfg.train_docs)])
    docs_val = tuple(bundle.val_docs[: int(cfg.val_docs)])
    docs_test = tuple(bundle.test_docs[: int(cfg.test_docs)])
    if int(cfg.train_docs) > len(docs_train):
        raise ValueError("bundle does not contain enough training docs for full-doc ladder stage")
    if int(cfg.val_docs) > len(docs_val):
        raise ValueError("bundle does not contain enough validation docs for full-doc ladder stage")
    if int(cfg.test_docs) > len(docs_test):
        raise ValueError("bundle does not contain enough test docs for full-doc ladder stage")

    seeds, device = _resolve_stage_device(cfg)
    target_scale = float(max(1, int(cfg.max_segments) - 1))
    train_doc_level = _prepare_doc_level_count_docs(
        docs_train,
        n_regimes=int(cfg.n_regimes),
        vocab_size=int(cfg.vocab_size),
        feature_mode=str(cfg.feature_mode),
    )
    val_doc_level = _prepare_doc_level_count_docs(
        docs_val,
        n_regimes=int(cfg.n_regimes),
        vocab_size=int(cfg.vocab_size),
        feature_mode=str(cfg.feature_mode),
    )
    test_doc_level = _prepare_doc_level_count_docs(
        docs_test,
        n_regimes=int(cfg.n_regimes),
        vocab_size=int(cfg.vocab_size),
        feature_mode=str(cfg.feature_mode),
    )
    doc_level_train_supervision = _doc_level_supervision_dataset(
        train_doc_level,
        split="train",
        target_scale=float(target_scale),
    )
    doc_level_val_supervision = _doc_level_supervision_dataset(
        val_doc_level,
        split="val",
        target_scale=float(target_scale),
    )
    doc_level_test_supervision = _doc_level_supervision_dataset(
        test_doc_level,
        split="test",
        target_scale=float(target_scale),
    )
    artifact_dir = Path(str(cfg.artifact_dir)) if str(cfg.artifact_dir).strip() else None
    doc_level_supervision_artifact = ""
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        doc_level_supervision_path = artifact_dir / "doc_level_supervision.json"
        doc_level_train_supervision.save(doc_level_supervision_path)
        doc_level_supervision_artifact = str(doc_level_supervision_path)

    doc_level_train, doc_level_val, doc_level_test, doc_level_fit = _fit_doc_level_baseline(
        config=cfg,
        seeds=seeds,
        target_scale=float(target_scale),
        device=device,
        train_docs=train_doc_level,
        val_docs=val_doc_level,
        test_docs=test_doc_level,
        train_supervision=doc_level_train_supervision,
        val_supervision=doc_level_val_supervision,
        test_supervision=doc_level_test_supervision,
    )
    (
        doc_level_ridge_train,
        doc_level_ridge_val,
        doc_level_ridge_test,
        doc_level_ridge_fit,
    ) = _fit_doc_level_ridge_baseline(
        train_docs=train_doc_level,
        val_docs=val_doc_level,
        test_docs=test_doc_level,
        train_supervision=doc_level_train_supervision,
        ridge_alpha=float(cfg.doc_level_ridge_alpha),
        tau=float(cfg.violation_tau),
    )
    doc_level_ridge_breakdown: Dict[str, Dict[str, Any]] = {}
    for order in tuple(int(x) for x in cfg.doc_level_ridge_breakdown_orders):
        label = {1: "unigram", 2: "bigram", 3: "trigram"}.get(int(order), f"{int(order)}gram")
        breakdown_train, breakdown_val, breakdown_test, breakdown_fit = _fit_doc_token_ngram_ridge_baseline(
            train_docs=docs_train,
            val_docs=docs_val,
            test_docs=docs_test,
            vocab_size=int(cfg.vocab_size),
            orders=(int(order),),
            ridge_alpha=float(cfg.doc_level_ridge_alpha),
            tau=float(cfg.violation_tau),
        )
        doc_level_ridge_breakdown[label] = {
            "train_root_mae": float(breakdown_train.root_mae),
            "val_root_mae": float(breakdown_val.root_mae),
            "test_root_mae": float(breakdown_test.root_mae),
            "selection_metric_name": str(breakdown_fit.selection_metric_name),
            "selection_metric_value": float(breakdown_fit.selection_metric_value),
        }

    doc_sequence_train, doc_sequence_val, doc_sequence_test, doc_sequence_fit = _fit_doc_sequence_baseline(
        config=cfg,
        seeds=seeds,
        device=device,
        train_docs=docs_train,
        val_docs=docs_val,
        test_docs=docs_test,
    )
    count_docs_train = _prepare_count_docs(
        docs_train,
        leaf_tokens=int(cfg.fixed_leaf_tokens),
        n_regimes=int(cfg.n_regimes),
        vocab_size=int(cfg.vocab_size),
        feature_mode=str(cfg.feature_mode),
    )
    count_docs_val = _prepare_count_docs(
        docs_val,
        leaf_tokens=int(cfg.fixed_leaf_tokens),
        n_regimes=int(cfg.n_regimes),
        vocab_size=int(cfg.vocab_size),
        feature_mode=str(cfg.feature_mode),
    )
    count_docs_test = _prepare_count_docs(
        docs_test,
        leaf_tokens=int(cfg.fixed_leaf_tokens),
        n_regimes=int(cfg.n_regimes),
        vocab_size=int(cfg.vocab_size),
        feature_mode=str(cfg.feature_mode),
    )
    rf_root_val = _eval_rf_root_baseline(
        count_docs_train,
        count_docs_val,
        seed=int(seeds["effective_model_seed"]),
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=int(cfg.rf_max_depth),
        min_samples_leaf=int(cfg.rf_min_samples_leaf),
    )
    rf_root_test = _eval_rf_root_baseline(
        count_docs_train,
        count_docs_test,
        seed=int(seeds["effective_model_seed"]) + 1_003,
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=int(cfg.rf_max_depth),
        min_samples_leaf=int(cfg.rf_min_samples_leaf),
    )
    train_target_diagnostics = {
        "n_docs": int(len(count_docs_train)),
        "n_unique": int(len({int(round(float(doc.root_count))) for doc in count_docs_train})),
        "is_constant": bool(
            len({int(round(float(doc.root_count))) for doc in count_docs_train}) <= 1
        ),
    }
    val_target_diagnostics = {
        "n_docs": int(len(count_docs_val)),
        "n_unique": int(len({int(round(float(doc.root_count))) for doc in count_docs_val})),
        "is_constant": bool(
            len({int(round(float(doc.root_count))) for doc in count_docs_val}) <= 1
        ),
    }
    test_target_diagnostics = {
        "n_docs": int(len(count_docs_test)),
        "n_unique": int(len({int(round(float(doc.root_count))) for doc in count_docs_test})),
        "is_constant": bool(
            len({int(round(float(doc.root_count))) for doc in count_docs_test}) <= 1
        ),
    }
    result = {
        "stage_name": str(stage.name),
        "description": str(stage.description),
        "source": "fresh_run",
        "runner_family": "full_doc_baselines_only",
        "reference_only": bool(stage.reference_only),
        "observed_token_profile": str(stage.observed_token_profile),
        "summary_json": str(stage_summary_path) if stage_summary_path is not None else "",
        "bundle_source": str(bundle_source),
        "generator_profile": str(cfg.generator_profile),
        "train_docs": int(cfg.train_docs),
        "val_docs": int(cfg.val_docs),
        "test_docs": int(cfg.test_docs),
        "state_dim": int(cfg.state_dim),
        "hidden_dim": int(cfg.hidden_dim),
        "n_epochs": int(cfg.n_epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "train_target_diagnostics": train_target_diagnostics,
        "val_target_diagnostics": val_target_diagnostics,
        "test_target_diagnostics": test_target_diagnostics,
        "degenerate_root_target_detected": bool(
            train_target_diagnostics["is_constant"]
            or val_target_diagnostics["is_constant"]
            or test_target_diagnostics["is_constant"]
        ),
        "root_count_support": _root_count_support(bundle),
        "doc_level_supervision_rows": int(len(doc_level_train_supervision.response_judgments)),
        "doc_level_supervision_artifact": str(doc_level_supervision_artifact),
        "doc_level_test_root_mae": float(doc_level_test.root_mae),
        "doc_level_val_root_mae": float(doc_level_val.root_mae),
        "doc_level_train_root_mae": float(doc_level_train.root_mae),
        "doc_level_ridge_test_root_mae": float(doc_level_ridge_test.root_mae),
        "doc_level_ridge_val_root_mae": float(doc_level_ridge_val.root_mae),
        "doc_level_ridge_train_root_mae": float(doc_level_ridge_train.root_mae),
        "doc_level_ridge_breakdown": doc_level_ridge_breakdown,
        "doc_sequence_test_root_mae": float(doc_sequence_test.root_mae),
        "doc_sequence_val_root_mae": float(doc_sequence_val.root_mae),
        "doc_sequence_train_root_mae": float(doc_sequence_train.root_mae),
        "doc_sequence_test_exact_match_rate": float(doc_sequence_fit.test_exact_match_rate),
        "doc_sequence_val_exact_match_rate": float(doc_sequence_fit.val_exact_match_rate),
        "doc_sequence_train_exact_match_rate": float(doc_sequence_fit.train_exact_match_rate),
        "doc_sequence_best_epoch": int(doc_sequence_fit.best_epoch),
        "doc_sequence_epochs_completed": int(doc_sequence_fit.epochs_completed),
        "doc_sequence_selection_metric_name": str(doc_sequence_fit.selection_metric_name),
        "doc_sequence_selection_metric_value": float(doc_sequence_fit.selection_metric_value),
        "doc_sequence_train_loss_final": float(doc_sequence_fit.train_loss_final),
        "doc_level_best_epoch": int(doc_level_fit.best_epoch),
        "doc_level_selection_metric_name": str(doc_level_fit.selection_metric_name),
        "doc_level_selection_metric_value": float(doc_level_fit.selection_metric_value),
        "doc_level_ridge_selection_metric_name": str(doc_level_ridge_fit.selection_metric_name),
        "doc_level_ridge_selection_metric_value": float(doc_level_ridge_fit.selection_metric_value),
        "rf_root_test_root_mae": float(rf_root_test.root_mae),
        "rf_root_val_root_mae": float(rf_root_val.root_mae),
        "learned_test_root_mae": float("nan"),
        "anchor_gap_to_ridge": float(doc_sequence_test.root_mae) - float(doc_level_ridge_test.root_mae),
    }
    result.update(
        {
            f"doc_sequence_{key}": value
            for key, value in official_fno_doc_sequence_provenance(
                objective_weights_active=False
            ).items()
        }
    )
    if stage_summary_path is not None:
        stage_summary_path.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def _extract_stage_result(
    *,
    stage: MarkovFullDocAnchorStageSpec,
    summary: OPSCountSummary,
    summary_json: str,
    bundle: MarkovOPSDataBundle | None,
    bundle_source: str,
    source: str,
) -> Dict[str, Any]:
    metrics = dict(summary.metrics or {})
    config = dict(summary.config or {})
    doc_sequence = dict(metrics.get("doc_sequence") or {})
    doc_sequence_train = dict(metrics.get("doc_sequence_train") or {})
    doc_sequence_val = dict(metrics.get("doc_sequence_val") or {})
    doc_sequence_training = dict(metrics.get("doc_sequence_training") or {})
    doc_level = dict(metrics.get("doc_level") or {})
    doc_level_ridge = dict(metrics.get("doc_level_ridge") or {})
    rf_root = dict(metrics.get("rf_root") or {})
    learned = dict(metrics.get("learned") or {})
    root_support = _root_count_support(bundle) if bundle is not None else {}
    result = {
        "stage_name": str(stage.name),
        "description": str(stage.description),
        "source": str(source),
        "reference_only": bool(stage.reference_only),
        "observed_token_profile": str(stage.observed_token_profile),
        "summary_json": str(summary_json),
        "bundle_source": str(bundle_source),
        "generator_profile": str(config.get("generator_profile", "")),
        "train_docs": int(config.get("train_docs", 0)),
        "val_docs": int(config.get("val_docs", 0)),
        "test_docs": int(config.get("test_docs", 0)),
        "state_dim": int(config.get("state_dim", 0)),
        "hidden_dim": int(config.get("hidden_dim", 0)),
        "n_epochs": int(config.get("n_epochs", 0)),
        "batch_size": int(config.get("batch_size", 0)),
        "lr": float(config.get("lr", 0.0)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "train_target_diagnostics": dict(config.get("train_target_diagnostics") or {}),
        "val_target_diagnostics": dict(config.get("val_target_diagnostics") or {}),
        "test_target_diagnostics": dict(config.get("test_target_diagnostics") or {}),
        "degenerate_root_target_detected": bool(
            config.get("degenerate_root_target_detected", False)
        ),
        "root_count_support": root_support,
        "doc_sequence_test_root_mae": _get_float(doc_sequence, "root_mae"),
        "doc_sequence_test_exact_match_rate": _get_float(
            doc_sequence_training,
            "test_exact_match_rate",
        ),
        "doc_sequence_val_root_mae": _get_float(doc_sequence_val, "root_mae"),
        "doc_sequence_train_root_mae": _get_float(doc_sequence_train, "root_mae"),
        "doc_sequence_best_epoch": int(doc_sequence_training.get("best_epoch", 0)),
        "doc_sequence_epochs_completed": int(
            doc_sequence_training.get("epochs_completed", 0)
        ),
        "doc_level_test_root_mae": _get_float(doc_level, "root_mae"),
        "doc_level_ridge_test_root_mae": _get_float(doc_level_ridge, "root_mae"),
        "rf_root_test_root_mae": _get_float(rf_root, "root_mae"),
        "learned_test_root_mae": _get_float(learned, "root_mae"),
        "anchor_gap_to_ridge": (
            _get_float(doc_sequence, "root_mae")
            - _get_float(doc_level_ridge, "root_mae")
        ),
    }
    result.update(
        {
            f"doc_sequence_{key}": value
            for key, value in official_fno_doc_sequence_provenance(
                objective_weights_active=False
            ).items()
        }
    )
    return result


def _rows_from_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage in list(payload.get("stages") or []):
        support = dict(stage.get("root_count_support") or {})
        row = {
            "stage_name": str(stage.get("stage_name", "")),
            "source": str(stage.get("source", "")),
            "observed_token_profile": str(stage.get("observed_token_profile", "")),
            "generator_profile": str(stage.get("generator_profile", "")),
            "degenerate_root_target_detected": bool(
                stage.get("degenerate_root_target_detected", False)
            ),
            "root_count_support_all_values": ",".join(
                str(int(x)) for x in list(support.get("all_values") or [])
            ),
            "doc_sequence_test_root_mae": _get_float(stage, "doc_sequence_test_root_mae"),
            "doc_sequence_test_exact_match_rate": _get_float(
                stage,
                "doc_sequence_test_exact_match_rate",
            ),
            "doc_level_test_root_mae": _get_float(stage, "doc_level_test_root_mae"),
            "doc_level_ridge_test_root_mae": _get_float(
                stage,
                "doc_level_ridge_test_root_mae",
            ),
            "rf_root_test_root_mae": _get_float(stage, "rf_root_test_root_mae"),
            "learned_test_root_mae": _get_float(stage, "learned_test_root_mae"),
            "anchor_gap_to_ridge": _get_float(stage, "anchor_gap_to_ridge"),
            "doc_sequence_backend_name": str(
                stage.get("doc_sequence_backend_name", "")
            ),
            "doc_sequence_backend_package": str(
                stage.get("doc_sequence_backend_package", "")
            ),
            "doc_sequence_backend_version": str(
                stage.get("doc_sequence_backend_version", "")
            ),
            "doc_sequence_operator_class": str(
                stage.get("doc_sequence_operator_class", "")
            ),
            "doc_sequence_operator_evidence_status": str(
                stage.get("doc_sequence_operator_evidence_status", "")
            ),
            "doc_sequence_theorem_relevance": bool(
                stage.get("doc_sequence_theorem_relevance", False)
            ),
            "doc_sequence_objective_weights_active": bool(
                stage.get("doc_sequence_objective_weights_active", False)
            ),
            "summary_json": str(stage.get("summary_json", "")),
            "bundle_source": str(stage.get("bundle_source", "")),
        }
        rows.append(row)
    return rows


def write_full_doc_anchor_ladder_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(str(key))
            fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render_full_doc_anchor_ladder_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Full-Doc Anchor Buildout",
        "",
        f"- preset: `{str(payload.get('preset', 'custom'))}`",
        "",
        "| stage | source | root-count support | doc-sequence root_mae | ridge root_mae | learned root_mae |",
        "|---|---|---|---:|---:|---:|",
    ]
    for stage in list(payload.get("stages") or []):
        support = dict(stage.get("root_count_support") or {})
        support_label = ",".join(str(int(x)) for x in list(support.get("all_values") or []))
        lines.append(
            "| "
            f"{str(stage.get('stage_name', ''))} | "
            f"{str(stage.get('source', ''))} | "
            f"{support_label or '-'} | "
            f"{_get_float(stage, 'doc_sequence_test_root_mae'):.6g} | "
            f"{_get_float(stage, 'doc_level_ridge_test_root_mae'):.6g} | "
            f"{_get_float(stage, 'learned_test_root_mae'):.6g} |"
        )
    return "\n".join(lines) + "\n"


def run_markov_full_doc_anchor_ladder(
    *,
    stage_specs: Sequence[MarkovFullDocAnchorStageSpec],
    output_dir: Path | None = None,
    use_cuda: bool = False,
    cuda_device: int | None = None,
    torch_threads: int = 1,
    skip_existing: bool = False,
    preset: str = "custom",
) -> Dict[str, Any]:
    if not stage_specs:
        raise ValueError("stage_specs must be non-empty")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stages").mkdir(parents=True, exist_ok=True)
    stages: List[Dict[str, Any]] = []
    for stage in stage_specs:
        stage_summary_path = (
            output_dir / "stages" / f"{stage.name}.json"
            if output_dir is not None
            else None
        )
        if (
            bool(skip_existing)
            and stage_summary_path is not None
            and stage_summary_path.exists()
        ):
            cached_payload = _load_cached_stage_result(stage_summary_path)
            cached_payload["source"] = "cached_run"
            cached_payload["summary_json"] = str(stage_summary_path)
            cached_payload["bundle_source"] = str(stage.bundle_path or cached_payload.get("bundle_source", ""))
            cached_payload.setdefault("stage_name", str(stage.name))
            cached_payload.setdefault("description", str(stage.description))
            cached_payload.setdefault("observed_token_profile", str(stage.observed_token_profile))
            cached_payload.setdefault("reference_only", bool(stage.reference_only))
            stages.append(cached_payload)
            continue

        if str(stage.summary_json).strip():
            path = Path(str(stage.summary_json))
            summary = _summary_from_json_path(path)
            bundle_path = Path(str(stage.bundle_path)) if str(stage.bundle_path).strip() else None
            bundle = (
                MarkovOPSDataBundle.load(bundle_path)
                if bundle_path is not None and bundle_path.exists()
                else None
            )
            stages.append(
                _extract_stage_result(
                    stage=stage,
                    summary=summary,
                    summary_json=str(path),
                    bundle=bundle,
                    bundle_source=str(bundle_path) if bundle_path is not None else "",
                    source="reference_summary",
                )
            )
            continue

        cfg = _stage_config_from_profile(
            stage,
            output_dir=output_dir,
            use_cuda=bool(use_cuda),
            cuda_device=cuda_device,
            torch_threads=int(torch_threads),
        )
        bundle, bundle_source = _load_or_build_bundle(
            stage=stage,
            cfg=cfg,
            output_dir=output_dir,
        )
        stages.append(
            _run_full_doc_baselines_stage(
                stage=stage,
                cfg=cfg,
                bundle=bundle,
                stage_summary_path=stage_summary_path,
                bundle_source=bundle_source,
            )
        )
    payload = {
        "simulation": "markov_full_doc_anchor_ladder",
        "preset": str(preset),
        "stages": stages,
    }
    return payload


__all__ = [
    "MarkovFullDocAnchorStageSpec",
    "render_full_doc_anchor_ladder_markdown",
    "resolve_markov_full_doc_anchor_ladder",
    "run_markov_full_doc_anchor_ladder",
    "write_full_doc_anchor_ladder_csv",
]
