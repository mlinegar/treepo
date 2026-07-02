# Changelog

## v0.1.0 — first release

First public version. There's no prior release to compare against.

Shipped public surface:

- Core experiment references, roles, manifests, honesty helpers, HLL sketches,
  local-law audit rows, objective metadata, and component-radius certificates.
- `treepo.fit()` and `treepo.methods.learning.fit()` as the single learning
  entrypoint over small built-in family runtimes.
- Unit-level `PreferenceDataset` records, task-state values, fine-tuning export
  views, and executable `ComposableStatistic` hooks for exact or learned
  tree-state methods.
- Built-in family routes for deterministic oracles, learnable constants,
  classical sketches, FNO, generic neural operators, provider-neutral LLM
  callables, and provider-neutral DSPy programs.
- `treepo-bench` run/check CLI, with small source-tree examples kept
  under `examples/`.
- `treepo[llm]` includes OpenAI-compatible and native Transformers
  dependencies for the unified chat/text-generation surface.

Known v0.1 limitations:

- `treepo.certificate` is a component-radius ledger. Lipschitz readout and
  measurement-error theorem terms are not first-class components yet; callers
  must include them in supplied radii when needed.
- HLL merge-learning `scalar_*` diagnostics are scalar readout checks, not
  state-level register-max Lean law certificates.
- Large optional application families are provided by downstream workspaces or
  packages. Large LDA recovery campaigns and full Manifesto/RILE workflows
  must be registered or run downstream.
- Source-tree examples are checkout fixtures only; they are not part of the
  public v0.1 API and are not required for wheel installs.

See [`README.md`](README.md) for the layout and
[`docs/training_defaults.md`](docs/training_defaults.md) for canonical defaults.
