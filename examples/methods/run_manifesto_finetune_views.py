#!/usr/bin/env python3
"""Export Manifesto/RILE fine-tuning rows from tree-based supervision.

The example keeps training out of treepo. It writes rows that embedding
trainers, SFT loops, DPO/reward-model loops, and GRPO/listwise loops can
consume downstream.
"""

from __future__ import annotations

from example_setup import (
    manifesto_finetune_fixture,
    manifesto_finetune_summary,
    parse_output_dir,
    write_json,
)


def main() -> int:
    from treepo.finetune import build_finetune_views, export_finetune_views, export_for_adapter
    from treepo.methods.preference import export_adapter_views
    from treepo.tasks.manifesto import replication_payload

    output_dir = parse_output_dir()
    train, tree_path, combined = manifesto_finetune_fixture(output_dir)
    views = build_finetune_views(combined)
    artifacts = export_finetune_views(combined, output_dir / "finetune")
    adapter_artifacts = export_adapter_views(export_for_adapter, combined, output_dir / "adapters")

    result = {
        "documents": replication_payload(train),
        "tree_records": str(tree_path),
        "counts": {name: len(rows) for name, rows in views.items()},
        "artifacts": artifacts,
        "adapters": adapter_artifacts,
        "summary": manifesto_finetune_summary(views),
        "previews": {name: rows[0] for name, rows in views.items() if rows},
    }
    out = output_dir / "manifesto_finetune_views_result.json"
    write_json(out, result)

    print(
        f"status=success family=manifesto_finetune_views "
        f"f_sft={result['summary']['f_sft_rows']} "
        f"g_sft={result['summary']['g_sft_rows']} "
        f"triplets={len(views['embedding_triplets'])} "
        f"ranked={len(views['embedding_ranked'])} "
        f"dpo={len(views['dpo'])} reward={len(views['reward'])} "
        f"grpo={len(views['grpo'])} output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
