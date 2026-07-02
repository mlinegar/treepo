#!/usr/bin/env python3
"""Central Manifesto/RILE replication example with unified preference data.

This is intentionally provider-neutral. By default it uses a tiny oracle
predictor so the example is runnable in package tests. Downstream manifesto
replications can swap in a real DSPy or LLM program through the same family
surface. Root and document-unit supervision is supplied through PreferenceDataset.
"""

from __future__ import annotations

from typing import Any

from example_setup import (
    ManifestoReplicationConfig,
    config_dict,
    load_example_config,
    manifesto_grid_cells,
    resolved_qsentence_sampling,
    run_manifesto_replication_cell,
    sample_desc,
    write_json,
)


def main() -> int:
    from treepo.tasks.manifesto import (
        manifesto_prompt_template,
        replication_payload,
    )

    output_dir, cfg = load_example_config(
        default_config="manifesto_replications.toml",
        config_cls=ManifestoReplicationConfig,
        cli_overrides=(("family", str),),
    )
    if cfg.preference_mode not in {"none", "scores", "pairwise", "ranked"}:
        raise ValueError("preference_mode must be one of: none, scores, pairwise, ranked")
    doc_unit_kind = str(cfg.doc_unit_kind or "qsentence")
    cells = manifesto_grid_cells(cfg)
    grid_mode = len(cells) > 1
    prompt_template = cfg.prompt_template or manifesto_prompt_template()
    grid_rows: list[dict[str, Any]] = []
    last_payload: dict[str, Any] | None = None
    for cell in cells:
        leaf_count = int(cell["leaf_unit_count"])
        mode = str(cell["preference_mode"])
        supervision_unit = str(cell["supervision_unit"])
        cell_dir = output_dir / f"{supervision_unit}_leaf_count_{leaf_count:03d}_{mode}"
        payload = run_manifesto_replication_cell(
            config=cfg,
            output_dir=cell_dir,
            leaf_unit_count=leaf_count,
            preference_mode=mode,
            prompt_template=prompt_template,
        )
        grid_rows.append(
            {
                "leaf_unit_count": leaf_count,
                "doc_unit_kind": doc_unit_kind,
                "preference_mode": mode,
                "preference_scope": cfg.preference_scope,
                "supervision_unit": supervision_unit,
                "status": payload["result"].status,
                "metrics": dict(payload["result"].metrics),
                "output_dir": str(cell_dir),
                "preference_counts": (payload["preference_artifacts"].get("counts") or {}),
                "sampling": payload["sampling_artifacts"]["summary"],
            }
        )
        last_payload = payload
    if last_payload is None:
        raise RuntimeError("manifesto grid produced no runnable cells")
    result = last_payload["result"]
    out = output_dir / "manifesto_replications_result.json"
    write_json(
        out,
        {
            "config": config_dict(cfg),
            "grid": grid_rows,
            "preferences": last_payload["preference_artifacts"],
            "replications": replication_payload(last_payload["eval_trees"]),
            "result": result.to_dict(),
            "sampling": last_payload["sampling_artifacts"],
        },
    )
    qsentence_sample_size, qsentence_sample_rate, _ = resolved_qsentence_sampling(cfg)
    doc_sample_desc = sample_desc(sample_size=cfg.doc_sample_size, sample_rate=cfg.doc_sample_rate)
    qsentence_sample_desc = sample_desc(sample_size=qsentence_sample_size, sample_rate=qsentence_sample_rate)
    grid_desc = f" grid_cells={len(grid_rows)}" if grid_mode else ""
    print(
        f"status={result.status} family={cfg.family} "
        f"docs={len(last_payload['eval_trees'])} train={len(last_payload['train'])} "
        f"doc_sample={doc_sample_desc} qsentence_sample={qsentence_sample_desc} "
        f"preferences={last_payload['preference_mode']} scope={cfg.preference_scope} "
        f"doc_unit={last_payload['doc_unit_kind']} "
        f"leaf_unit_count={last_payload['leaf_unit_count']}{grid_desc} "
        f"mae={result.metrics.get('internal_f_mae')} output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
