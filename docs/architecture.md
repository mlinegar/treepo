# treepo v0.1.0 Architecture

`treepo` is organized by intent rather than by the historical script layout.

## Layers

- `treepo.core`: small dependency-free experiment primitives: refs, roles, sampling plans, and canonical sidecars.
- `treepo.sketches`: sketch protocols and adapters. Optional third-party sketch backends should fail lazily with an extra hint.
- `treepo.bench`: reproducible simulations, suite builders, result IO, and reports.
- `treepo.runtime`: benchmark adapters and runtime task helpers for LongBench/RULER-style evaluation.
- `treepo.llm`: OpenAI-compatible chat/embedding helpers and future batching clients behind `treepo[llm]`.
- `treepo.training`: experiment methods with `train`, `evaluate`, and `predict` wrappers around native PyTorch/DSPy/sklearn code.
- `treepo.tasks`: minimal task-specific assets, starting with Manifesto/RILE constants and examples.

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

Internal runtime surfaces may still be chat, embedding, or operator endpoints; those are implementation details.

## Migration Rule

Root workspace scripts are not copied wholesale. Each candidate is classified in `migration_inventory.yaml` before moving into package code:

- `package_module`: importable implementation module
- `cli_command`: public `treepo-bench` command
- `compat_shim`: compatibility layer kept intentionally thin
- `exclude_legacy`: workspace-only or historical utility

The release gate is:

```bash
treepo-bench check inventory --json
treepo-bench check hygiene --json
```
