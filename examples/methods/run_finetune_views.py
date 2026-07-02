#!/usr/bin/env python3
"""Export embedding and LLM fine-tuning rows from one preference dataset.

The example is trainer-neutral. It does not call model services and does not
instantiate sentence-transformers, TRL, PEFT, or Accelerate objects.
"""

from __future__ import annotations

from example_setup import parse_output_dir, toy_finetune_preferences, write_json


def main() -> int:
    from treepo.finetune import build_finetune_views, export_finetune_views, export_for_adapter
    from treepo.methods.preference import export_adapter_views

    output_dir = parse_output_dir()
    preferences = toy_finetune_preferences()
    views = build_finetune_views(preferences)
    artifacts = export_finetune_views(preferences, output_dir / "finetune")
    adapter_artifacts = export_adapter_views(export_for_adapter, preferences, output_dir / "adapters")

    result = {
        "counts": {name: len(rows) for name, rows in views.items()},
        "artifacts": artifacts,
        "adapters": adapter_artifacts,
        "previews": {name: rows[0] for name, rows in views.items() if rows},
    }
    out = output_dir / "finetune_views_result.json"
    write_json(out, result)

    print(
        f"status=success family=finetune_views "
        f"embedding_pairs={len(views['embedding_pairs'])} "
        f"embedding_triplets={len(views['embedding_triplets'])} "
        f"embedding_ranked={len(views['embedding_ranked'])} "
        f"sft={len(views['sft'])} dpo={len(views['dpo'])} "
        f"reward={len(views['reward'])} grpo={len(views['grpo'])} "
        f"output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
