#!/usr/bin/env python3
"""Export Manifesto reward-mechanism views from one preference dataset shape.

The example is trainer-neutral. It builds scoped Manifesto preferences and
exports supervised, DPO, reward-model, and GRPO projections for:

* root-only document rewards,
* qsentence-only state rewards,
* combined root + qsentence rewards.
"""

from __future__ import annotations

from example_setup import (
    ManifestoRewardMechanismConfig,
    config_dict,
    load_example_config,
    write_json,
)

from treepo.tasks.sampling_artifacts import write_sampling_artifacts


def main() -> int:
    from treepo.tasks.manifesto import (
        export_manifesto_reward_views,
        make_manifesto_replication_trees,
        manifesto_document_unit_sampling_rows,
        sample_manifesto_replication_trees,
    )

    output_dir, cfg = load_example_config(
        default_config="manifesto_reward_mechanisms.toml",
        config_cls=ManifestoRewardMechanismConfig,
    )
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

    rows = export_manifesto_reward_views(
        output_dir=output_dir / "reward_views",
        trees=train,
        preference_scopes=cfg.preference_scopes,
        preference_modes=cfg.preference_modes,
        sample_size=cfg.qsentence_sample_size,
        sample_rate=cfg.qsentence_sample_rate,
        seed=int(cfg.qsentence_sample_seed),
        export_formats=cfg.export_formats,
    )

    out = output_dir / "manifesto_reward_mechanisms_result.json"
    write_json(
        out,
        {
            "config": config_dict(cfg),
            "sampling": sampling_artifacts,
            "cells": rows,
        },
    )
    print(
        f"status=success family=manifesto_reward_mechanisms "
        f"docs={len(train)} qsentences={sampling_artifacts['summary']['qsentences']['population_count']} "
        f"cells={len(rows)} output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
