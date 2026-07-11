# Per-cell `results.json` (results_version 0.1)

Every `fit()` run writes `results.json` to its `output_dir`, next to the run
manifest. It is the one artifact a grid cell contributes to cross-cell
comparison tooling (the ThinkingTrees W-ledger scripts). Writer:
`treepo.methods._results.write_results_json`.

## Blocks

- **`cell`** — what was run: family, schedule, seed, axis, named supervision
  level + resolved weights, full grid-axes provenance (doc_gold selection,
  label mix + attach report), objective, final stage label.
- **`metrics`** — `pooled_across_dimensions: false` always;
  `splits.<split>` carries:
  - `external` / `internal`: `n`, `pearson_r`, `mae_native`, and
    `normalized_abs_error` = mean |ŷ−y| / (b−a) — the W1 `R_j` — with
    `scale_bounds` and `scale_bounds_source` recorded (`observed_gold_range`
    until a task supplies native bounds);
  - `per_dimension`: unpooled per-dimension metrics (pooling across
    dimensions inflates Pearson and is banned);
  - `sim`: `theta_first_regime_accuracy`, `theta_last_regime_accuracy`,
    `contextual_mae` — the standing pairing, null when the cell has no sim
    channel but always present.
- **`local_laws`** — the evidence artifact's per-law summary
  (`summary`, `by_law_kind`, `source`).
- **`cost`** — three components, reported separately, never blended:
  - `label_cost`: gold doc labels consumed (pinned doc_gold count), node
    label source, gold vs distilled node-label counts, leaf/merge row counts;
  - `one_time_compute`: fit wall seconds, train tree count, iterations;
  - `marginal_inference`: eval prediction count;
  - `resummary_ops`: `{count, population}` — recorded even when zero
    (`empty_by_construction`), so an empty deployed-C2 stratum stays visible
    instead of reading as a pass.
- **`paired_rows`** — pointers to the per-document
  `prediction_records/iter_*_post_eval.jsonl` files plus the field mapping
  (`key=tree_id`, `prediction=prediction_scalar`, `gold=expert_score`,
  `teacher=teacher_score`, `split=split`) that paired-Δ / bootstrap tooling
  ingests.
