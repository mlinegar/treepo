# Changelog

## v0.1.0 — first release

First public version.

Shipped public surface:

- `treepo.fit()` and `treepo.methods.learning.fit()` as the single learning
  entrypoint over the built-in family runtimes.
- Seven built-in families: deterministic oracles, learnable constants,
  classical sketches, FNO, generic neural operators, provider-neutral LLM
  callables, and provider-neutral DSPy programs. Downstream packages register
  additional runtimes with `treepo.methods.families.register_family(...)`.
- Unit-level `PreferenceDataset` records, JSONable task states, tree records
  with validation and summaries, fine-tuning export views, and executable
  `ComposableStatistic` hooks for exact or learned tree-state methods.
- Local-law training objectives (`treepo.training.local_law`), local-law audit
  rows, objective metadata, and component-radius certificates
  (`treepo.certificate`).
- The `treepo-bench` CLI: `run` for single experiments with JSON/CSV output,
  `check {inventory,hygiene,release}` for package gates.
- Extras: `bench` (YAML config IO), `sketches` (datasketches adapters), `llm`
  (the requests-based OpenAI-compatible client layer), and `all`.
- Source-tree examples under `examples/bench` and `examples/methods`.

Scope notes for v0.1:

- `treepo.certificate` is a component-radius ledger; callers supply Lipschitz
  readout and measurement-error terms in their radii where a bound needs them.
- Large application workflows (full LDA recovery campaigns, Manifesto/RILE
  training) run in downstream packages that register their families here.
- The wheel ships the `treepo` package; examples live in the source checkout.

See [`README.md`](README.md) for the layout and
[`docs/training_defaults.md`](docs/training_defaults.md) for canonical defaults.
