# treepo Architecture

`treepo` is organized by package intent.

## Layers

- `treepo.methods` plus the top-level value modules: the package's center. See below.
- `treepo.bench.sketches`: the sketch adapter protocol and tree reducer. Optional third-party sketch backends load lazily and name the extra to install.
- `treepo.bench`: benchmark runs, result IO, and release checks. `treepo.bench.classical_sketches` is the comparison benchmark that runs adapters from `treepo.bench.sketches` over shared item streams.
- `treepo.llm`: OpenAI-compatible chat/embedding helpers behind `treepo[llm]`.
  vLLM, SGLang, hosted compatible APIs, and compatible local servers use the
  same `/v1` client; direct local runtimes such as Transformers plug in through
  `predict_fn`.
- `treepo.training`: torch local-law tensor helpers layered on `treepo.local_law`; richer trainers register from downstream packages.
- `treepo.tasks`: small task-specific assets, starting with Manifesto/RILE constants and examples.

## The Methods Layer

`treepo.fit(...)` is the single public learning surface. It normalizes a
mapping spec, resolves one family runtime from the registry in
`treepo.methods.families`, runs the alternating f/g loop in
`treepo.methods.runtime`, and assembles a `FitResult` with metrics, artifacts,
history, and a manifest.

Seven families are built in: `oracle`, `learnable_constant`,
`classical_sketch`, `neural_operator`, `fno`, `llm`, and `dspy`. Downstream
packages register additional runtimes with
`treepo.methods.families.register_family(...)` against the `FamilyRuntime`
protocol in `treepo.methods.contracts`. Wrappers own the training loop and
call `module.train()` / `module.eval()` internally, so every family presents
the same `train_f` / `train_g` / `score_roots_with_f` surface to the runtime.

`treepo.methods.preference` holds the unit-level supervision boundary:
`Candidate`, `PreferenceRecord`, and `PreferenceDataset`, with one canonical
Hugging Face `DatasetDict` shape and generic/supervised/DPO/reward/GRPO
projection exports. See [`preference_data.md`](preference_data.md) for
root-level and node-level loading patterns.

The top-level value modules carry the package's data shapes and diagnostics:

- `treepo.state` — `TaskState`, the JSONable state produced by `g` and read by `f`.
- `treepo.tree` — `TreeNode` / `TreeRecord`, the minimal labeled tree artifact.
- `treepo.statistic` — the executable `ComposableStatistic` protocol for encode/merge/readout.
- `treepo.local_law` — canonical, Lean-aligned scalar C1/C2/C3 row arithmetic and audit summaries.
- `treepo.evidence` — the unified per-run evidence artifact (see [`docs/evidence.md`](evidence.md)).
- `treepo.certificate` — the component-radius certificate ledger.
- `treepo.objective` — objective metadata for manifests and evidence.
- `treepo.sampling` — design-propensity sampling helpers.
- `treepo.artifacts` — canonical run-artifact bundles.
- `treepo.finetune` — trainer-neutral embedding and LLM fine-tuning export views.
- `treepo.common` — small shared utilities such as `stable_digest`.

## Role Vocabulary

Public role metadata follows the paper language:

- `scorer`: practical task scorer `f`
- `summarizer`: summarizer `g`
- `oracle`: trusted target/evaluator `f*` or benchmark labels
- `embedder`: vector evidence mechanism
- `state_model`: learned or deterministic state realization

Internal method surfaces are implementation details — chat, embedding, or
operator endpoints all map onto the same public roles.

## Package Inventory

`inventory.yaml` records the package boundary:

- `package`: importable implementation module
- `cli`: public `treepo-bench` command
- `shim`: thin package shim
- `outside`: code owned by downstream packages
- `extension`: optional family or backend registered by another package

Release checks:

```bash
treepo-bench check inventory --json
treepo-bench check hygiene --json
treepo-bench check release --json
```
