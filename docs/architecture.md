# treepo v0.1.0 Architecture

`treepo` is organized by package intent.

## Layers

- `treepo.core`: small dependency-free experiment primitives: refs, roles, sampling plans, and canonical sidecars.
- `treepo.bench.sketches`: sketch protocols and adapters. Optional third-party sketch backends should fail lazily with an extra hint.
- `treepo.bench`: benchmark runs, result IO, and release checks.
- `treepo.llm`: OpenAI-compatible chat/embedding helpers and future batching clients behind `treepo[llm]`.
- `treepo.training`: lightweight training protocols and torch local-law helpers; richer trainers register from downstream packages.
- `treepo.tasks`: small task-specific assets, starting with Manifesto/RILE constants and examples.

## Experiment Contract

One public noun is used: experiment. An `ExperimentContext` records:

- `experiment_id`
- `BenchmarkRef`
- `MethodRef`
- `SamplingPlan`
- canonical sidecars: `experiment_manifest.json`, `experiment_status.json`, `artifacts.json`, and `results.jsonl`

Methods may expose `train`, `evaluate`, and `predict`. Raw PyTorch modules are not experiment methods; wrappers own the training loop and call `module.train()` / `module.eval()` internally.

## Role Vocabulary

Public role metadata follows the paper language:

- `scorer`: practical task scorer `f`
- `summarizer`: summarizer `g`
- `oracle`: trusted target/evaluator `f*` or benchmark labels
- `embedder`: vector evidence mechanism
- `state_model`: learned or deterministic state realization

Internal method surfaces may still be chat, embedding, or operator endpoints; those are implementation details.

## Package Inventory

`inventory.yaml` records the package boundary:

- `package`: importable implementation module
- `cli`: public `treepo-bench` command
- `shim`: thin package shim
- `outside`: code that belongs outside the package
- `extension`: optional family or backend registered by another package

Release checks:

```bash
treepo-bench check inventory --json
treepo-bench check hygiene --json
treepo-bench check release --json
```
