#!/usr/bin/env python3
"""Full Manifesto/RILE walkthrough: sampling, fit, evidence, and rewards."""

from __future__ import annotations

from typing import Any

from example_setup import (
    ManifestoEndToEndConfig,
    config_dict,
    load_example_config,
    write_json,
)
from treepo.tasks.sampling_artifacts import write_sampling_artifacts


def main() -> int:
    from treepo import fit
    from treepo.artifacts import write_artifact_bundle
    from treepo.methods.preference import export_preference_records, summarize_preference_views
    from treepo.tasks.manifesto import (
        export_manifesto_reward_views,
        make_manifesto_preferences,
        make_manifesto_replication_trees,
        manifesto_document_unit_sampling_rows,
        manifesto_oracle_predict_fn,
        manifesto_prompt_template,
        manifesto_tree_records,
        replication_payload,
        sample_manifesto_replication_trees,
    )

    output_dir, cfg = load_example_config(
        default_config="manifesto_end_to_end.toml",
        config_cls=ManifestoEndToEndConfig,
    )
    if cfg.fit_preference_mode not in {"none", "scores", "pairwise", "ranked"}:
        raise ValueError("fit_preference_mode must be one of: none, scores, pairwise, ranked")

    train_population = make_manifesto_replication_trees(
        split="train",
        leaf_unit_count=int(cfg.leaf_unit_count),
        doc_unit_kind=str(cfg.doc_unit_kind),
    )
    train, document_sampling_rows = sample_manifesto_replication_trees(
        train_population,
        sample_size=cfg.doc_sample_size,
        sample_rate=cfg.doc_sample_rate,
        seed=int(cfg.doc_sample_seed),
    )
    eval_trees = make_manifesto_replication_trees(
        split="test",
        leaf_unit_count=int(cfg.leaf_unit_count),
        doc_unit_kind=str(cfg.doc_unit_kind),
    )
    qsentence_sampling_rows = manifesto_document_unit_sampling_rows(
        train,
        sample_size=cfg.qsentence_sample_size,
        sample_rate=cfg.qsentence_sample_rate,
        seed=int(cfg.qsentence_sample_seed),
    )
    sampling_artifacts = write_sampling_artifacts(
        output_dir,
        document_rows=document_sampling_rows,
        qsentence_rows=qsentence_sampling_rows,
    )

    train_tree_records = manifesto_tree_records(train)

    fit_preferences = None
    fit_preference_artifacts: dict[str, Any] = {}
    fit_optimizer_views: dict[str, Any] = {}
    if cfg.fit_preference_mode != "none":
        fit_preferences = make_manifesto_preferences(
            train,
            mode=str(cfg.fit_preference_mode),
            scope=str(cfg.fit_preference_scope),
            sample_size=cfg.qsentence_sample_size,
            sample_rate=cfg.qsentence_sample_rate,
            seed=int(cfg.qsentence_sample_seed),
        )
        fit_preference_artifacts = {
            "mode": str(cfg.fit_preference_mode),
            "scope": str(cfg.fit_preference_scope),
            **export_preference_records(fit_preferences, output_dir / "fit_preference"),
        }
        fit_optimizer_views = summarize_preference_views(fit_preferences)

    backend: dict[str, Any] = {
        "output_dir": str(output_dir / "fit"),
        "model": str(cfg.model),
        "prompt_template": cfg.prompt_template or manifesto_prompt_template(),
        "min_score": -100.0,
        "max_score": 100.0,
    }
    if bool(cfg.use_oracle_predictor):
        backend["predict_fn"] = manifesto_oracle_predict_fn

    fit_config: dict[str, Any] = {
        "family": str(cfg.family),
        "train_data": train,
        "eval_data": eval_trees,
        "backend_config": backend,
        "axis": {
            "max_iterations": int(cfg.max_iterations),
            "axis_value": int(cfg.leaf_unit_count),
            "axis_kind": "leaf_unit_count",
        },
    }
    if fit_preferences is not None:
        fit_config["preference_data"] = fit_preferences
    result = fit(fit_config)
    evidence = dict(result.artifacts.get("evidence") or {})
    artifact_bundle = write_artifact_bundle(
        output_dir / "artifact_bundle",
        trees=train_tree_records,
        preference_data=fit_preferences,
        metadata={
            "example": "manifesto_end_to_end",
            "fit_preference_mode": str(cfg.fit_preference_mode),
            "fit_preference_scope": str(cfg.fit_preference_scope),
        },
    )

    reward_views = export_manifesto_reward_views(
        output_dir=output_dir / "reward_views",
        trees=train,
        preference_scopes=cfg.reward_scopes,
        preference_modes=cfg.reward_modes,
        sample_size=cfg.qsentence_sample_size,
        sample_rate=cfg.qsentence_sample_rate,
        seed=int(cfg.qsentence_sample_seed),
        export_formats=cfg.export_formats,
    )

    evidence_path = output_dir / "evidence.json"
    result_path = output_dir / "manifesto_end_to_end_result.json"
    write_json(evidence_path, evidence)
    write_json(
        result_path,
        {
            "config": config_dict(cfg),
            "artifact_bundle": artifact_bundle,
            "evidence": evidence,
            "files": {
                "evidence": str(evidence_path),
                "result": str(result_path),
                "fit_manifest": result.manifest_path,
            },
            "fit_optimizer_views": fit_optimizer_views,
            "fit_preferences": fit_preference_artifacts,
            "replications": {
                "train": replication_payload(train),
                "eval": replication_payload(eval_trees),
            },
            "result": result.to_dict(),
            "reward_views": reward_views,
            "sampling": sampling_artifacts,
        },
    )

    fit_pref = (
        f"{cfg.fit_preference_mode}/{cfg.fit_preference_scope}"
        if cfg.fit_preference_mode != "none"
        else "none"
    )
    print(
        f"status={result.status} family={cfg.family} "
        f"docs={len(train)} eval={len(eval_trees)} "
        f"qsentences={sampling_artifacts['summary']['qsentences']['population_count']} "
        f"fit_preferences={fit_pref} reward_cells={len(reward_views)} "
        f"mae={result.metrics.get('internal_f_mae')} output={result_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
