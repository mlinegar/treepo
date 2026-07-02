#!/usr/bin/env python3
"""Minimal optimizer-view and DSPy prompt example.

One ``PreferenceDataset`` supplies:

* supervised f/g rows for ``treepo.fit(..., family="dspy")``;
* DPO rows for pairwise optimizers;
* reward-model rows with scores;
* grouped GRPO rows for listwise/ranked optimizers.

The example is trainer-neutral and does not call an external model service.
"""

from __future__ import annotations

from example_setup import (
    parse_output_dir,
    toy_dspy_fit_config,
    toy_optimizer_trees_and_preferences,
    write_json,
)


def main() -> int:
    from treepo import fit
    from treepo.artifacts import write_artifact_bundle
    from treepo.methods.preference import export_preference_records, summarize_preference_views

    output_dir = parse_output_dir()
    train_trees, preferences = toy_optimizer_trees_and_preferences()

    preference_artifacts = export_preference_records(preferences, output_dir / "preference")
    bundle = write_artifact_bundle(
        output_dir / "bundle",
        trees=train_trees,
        preference_data=preferences,
        metadata={"example": "preference_optimizer_views"},
    )

    fit_result = fit(
        toy_dspy_fit_config(
            output_dir=output_dir,
            train_trees=train_trees,
            preferences=preferences,
        )
    )

    result = {
        "status": fit_result.status,
        "bundle": bundle,
        "fit": fit_result.to_dict(),
        "optimizer_views": summarize_preference_views(preferences),
        "preference_artifacts": preference_artifacts,
    }
    result_path = output_dir / "preference_optimizer_views_result.json"
    write_json(result_path, result)

    print(
        f"status={fit_result.status} family=dspy "
        f"units={preference_artifacts['counts']['units']} "
        f"dpo={preference_artifacts['counts']['dpo']} "
        f"reward={preference_artifacts['counts']['reward']} "
        f"grpo={preference_artifacts['counts']['grpo']} "
        f"output={result_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
