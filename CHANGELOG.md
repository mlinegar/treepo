# Changelog

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
