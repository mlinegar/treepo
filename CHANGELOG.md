# Changelog

## Unreleased — on `main` since the v0.2.0 tag

The package version still reads 0.2.0. Additions since the tag focus on
running real supervision grids through `fit()`:

- Supervision-grid axes as spec fields: `doc_gold_n` (nested, pinned
  per-seed document subsets), `local_label_mix` (`none` / `gold_fraction` /
  `llm_distilled`), and `seed`, plus a grid-cell expander
  (`treepo.methods._grid_axes`).
- Labeled-bundle loader (`treepo.bundles.load_labeled_tree_bundle`) that reads
  ThinkingTrees labeled-tree bundles into `TreeRecord`s with pinned splits;
  the on-disk contract is in [`docs/bundle_contract.md`](docs/bundle_contract.md).
- Per-node supervision through `fit()`: named supervision levels
  (`default` / `root` / `leaf` / `node` / `mix`), explicit root/leaf/merge
  weights, trace-aligned node targets, a `local_law_weight` shorthand for the
  canonical convex objective, and an opt-in per-tree convex loss
  (`per_tree_loss_lambda`).
- Additive leaf-to-document regime: `root_readout="leaf_mean"` on the
  neural-operator families, with a rollup-identity check
  (`treepo.bundles.leaf_rollup_report`).
- Distillation lane (`treepo.distilled`): cached teacher node scores train a
  student through `fit()` with no LLM client; placeholder caches are rejected.
- Per-cell `results.json` beside the run manifest, with unpooled metrics and
  separate label / compute / inference cost entries
  ([`docs/results_schema.md`](docs/results_schema.md)).
- FNO family: the embedding-coordinate `f`/`g` model with identity-style
  initialization, exact gradient accumulation (`micro_batch_size`), chunked
  scoring (`eval_batch_size`), and leaf encoding that skips padding to cut
  memory.
- README rewritten around the paper, with an offline quickstart
  (`examples/quickstart.py`).

## v0.2.0 — unified `fit()` surface

Added:

- Executed objectives: `fit()` resolves the `ObjectiveSpec` once and hands it
  to the family through `configure_objective`. The neural-operator families
  train on the convex root + local-law objective; a family that cannot execute
  a law-bearing objective raises instead of recording it as provenance only.
- `treepo.schedule`: one canonical definition of merge order, children,
  depths, and folds, shared by every family.
- Warmstart: neural-operator artifacts save their weights (`weights_path`),
  and passing them back through `initial_artifacts` resumes the trained model.
- Local-law audit rows from every built-in family, with law auditing on by
  default for the `llm` and `dspy` routes. Law kinds accept the paper, Lean
  numbering, and Lean channel spellings from one alias table.
- `sklearn_lda` family.
- `treepo.identification`: additive root-sensitivity and information weights
  for partially observed trees.
- Two-channel error certificates
  (`treepo.certificate.build_two_channel_error_certificate`,
  `treepo.local_law.triangle_local_law_residual_from_audit`) with
  common-mechanism and conditional-average envelope adapters; see
  [`docs/evidence.md`](docs/evidence.md).
- LLM client: `OpenAICompatibleChatClient` and `build_chat_client`. The `llm`
  family calls OpenAI-compatible `/v1` servers (vLLM, SGLang, hosted APIs)
  directly or uses an injected `predict_fn`; `dspy` takes the same server
  settings. Example: `examples/methods/run_llm_backends.py`.
- [`docs/preference_data.md`](docs/preference_data.md) documents the
  `PreferenceDataset` boundary.

Changed:

- The core install is numpy-only. `torch`/`neuraloperator`,
  `scikit-learn`/`scipy`, and `datasets` move to the `torch`, `lda`, and `hf`
  extras (all included in `all`); a missing extra fails at first use and names
  the extra.
- `fit()` trains by default (one `f` pass and one `g` pass); pass
  `axis={"max_iterations": 0}` for an evaluation-only run.
- Without an `output_dir`, `treepo.methods.learning.fit()` now writes to
  `outputs/treepo_fit`, matching `treepo.fit()`, instead of a temporary
  directory.

## v0.1.1 — first release

Shipped public surface:

- `treepo.fit()` and `treepo.methods.learning.fit()` as the single learning
  entrypoint over the built-in family runtimes.
- Seven built-in families: deterministic oracles, learnable constants,
  classical sketches, FNO, generic neural operators, provider-neutral LLM
  callables, and provider-neutral DSPy programs. Downstream packages register
  additional runtimes with `treepo.methods.families.register_family(...)`.
- Unit-level `PreferenceDataset` records, JSONable task states, tree records
  with validation and summaries, fine-tuning export views, and executable
  `ComposableStatistic` hooks — including per-node `f` readouts
  (`node_readouts`) over the merge trace.
- Local-law training objectives (`treepo.training.local_law`) with the
  root-at-depth-0 `gamma^depth` convention, local-law audit rows, uniform
  node-audit designs (`treepo.sampling.sample_node_audit` /
  `apply_node_audit`), objective metadata, and component-radius certificates
  (`treepo.certificate`).
- The tree visualization (`treepo.viz.write_tree_visualization_html`): one
  self-contained HTML file per run with sampled-node markers, gold and
  prediction labels, text and `g`-state summaries, per-node readouts,
  local-law losses on the synthesized merge tree, and audit, certificate,
  and tradeoff panels.
- The `TradeoffCurve` record (`treepo.methods.tradeoff`): the named
  error-vs-`leaf_unit_count` artifact, built from grid rows, written as
  JSON+CSV, rendered by the visualization.
- Fixture record converters (`markov_tree_records`, `lda_tree_records`,
  `hll_tree_records`) with exact per-leaf gold labels and one shared
  `tree_id` convention across DGPs.
- The `treepo-bench` CLI: `run` for single experiments with JSON/CSV output,
  `check {inventory,hygiene,release}` for package gates.
- Extras: `bench` (YAML config IO), `sketches` (datasketches adapters), `llm`
  (the requests-based OpenAI-compatible client layer), and `all`.
- Source-tree examples under `examples/bench` and `examples/methods`,
  including five reference visualization views.

Scope notes:

- `treepo.certificate` is a component-radius ledger; callers supply Lipschitz
  readout and measurement-error terms in their radii where a bound needs them.
- Large application workflows (full LDA recovery campaigns, Manifesto/RILE
  training) run in downstream packages that register their families here.
- The wheel ships the `treepo` package; examples live in the source checkout.

See [`README.md`](README.md) for the layout and
[`docs/training_defaults.md`](docs/training_defaults.md) for canonical defaults.
