# LLM Guide For treepo

Last audited: 2026-07-02.

This guide is for LLM/code agents working in the standalone `treepo` package. It is the operational companion to `README.md`, `docs/architecture.md`, `docs/boundary.md`, and `docs/training_defaults.md`.

## Start Here

1. Read `AGENTS.md`, then this file.
2. Check `git status --short` before editing. The worktree may already contain user changes.
3. Use `rg` for source search and inspect nearby tests before changing behavior.
4. Keep the package boundary small: `treepo` is a public package; the paper workspace, model-serving launchers, and publication grids live in downstream repos.
5. Prefer public contracts already in `src/treepo/` over adding new ad hoc data shapes.

## Core Rule

`treepo` owns stable C-TreePO interfaces: JSONable task states, labeled tree records, preference records, local-law rows, evidence artifacts, a small `treepo.fit(...)` learning surface, and small runnable examples.

Downstream workspaces own large datasets, model-serving fleets, concrete DSPy programs, trainer applications, long publication campaigns, and domain-specific orchestration. Those systems adapt to `treepo` contracts and register through the public APIs.

## Repo Map

| Path | Role |
| --- | --- |
| `src/treepo/__init__.py` | Lazy public exports. Keep top-level import light. |
| `src/treepo/learning.py` | Package-level `treepo.fit(...)` wrapper. |
| `src/treepo/methods/contracts.py` | Stable fit and family contracts. |
| `src/treepo/methods/learning.py` | Single internal fit loop over a `FamilyRuntime`. |
| `src/treepo/methods/families.py` | Small built-in family registry and extension boundary. |
| `src/treepo/methods/runtime.py` | Alternating f/g runtime and split metrics. |
| `src/treepo/local_law.py` | Canonical C1/C2/C3 row arithmetic and audit summaries. |
| `src/treepo/state.py` | `TaskState` and JSON conversion helpers. |
| `src/treepo/tree.py` | Minimal `TreeNode` / `TreeRecord` artifact shape. |
| `src/treepo/statistic.py` | Public composable statistic protocol. |
| `src/treepo/evidence.py` | Unified evidence artifact builder. |
| `src/treepo/finetune.py` | Trainer-neutral embedding and LLM fine-tuning views. |
| `src/treepo/bench/` | `treepo-bench` runner, benchmark config IO, checks. |
| `src/treepo/llm/` | Optional client-side LLM and embedding helpers; the deploying package owns server orchestration. |
| `src/treepo/training/` | Torch tensor layer over `treepo.local_law` training rows. |
| `src/treepo/tasks/manifesto/` | Small Manifesto/RILE fixture/state helpers. |
| `examples/` | Small runnable examples only. |
| `inventory.yaml` | Package boundary inventory checked by release gates. |

## Public Data Shapes

Use these before inventing a new structure:

- `TaskState`: JSONable state produced by `g` and read by `f`.
- `TreeNode` / `TreeRecord`: package-owned tree artifact representation.
- `Candidate`, `PreferenceRecord`, `PreferenceDataset`: the unit-level supervision and preference boundary.
- `LocalLawAuditRow`: theorem-facing C1/C2/C3 row with observed mask, propensity, node weight, depth, and optional oracle loss.
- `ObjectiveSpec`: objective metadata for manifests and evidence.
- `FitResult`: uniform package result with metrics, artifacts, history, summary, and manifest path.

## How Fit Works

The public call is:

```python
import treepo

result = treepo.fit(
    {
        "family": "neural_operator",
        "train_data": train_trees,
        "eval_data": eval_trees,
        "preference_data": preferences,
        "backend_config": {"operator_kind": "fno"},
        "axis": {"max_iterations": 2},
    }
)
```

The path is:

1. `treepo.fit(...)` normalizes a mapping into `FitConfig`.
2. `treepo.methods.contracts.CTreePOLearningSpec` receives the public spec.
3. `treepo.methods.families.resolve_family(...)` builds a `FamilyRuntime`, unless `backend_config["family_runtime"]` injects one for tests or downstream adapters.
4. `treepo.methods._grid_axes.apply_grid_axes(...)` pins the `doc_gold_n` document subset and resolves the `local_label_mix` node-label source before training (see `docs/training_defaults.md` for the axis surface and defaults).
5. `treepo.methods.runtime.run_alternating_family(...)` alternates f/g training and evaluates splits.
6. `treepo.methods._fit_result.build_result(...)` writes manifests, prediction rows, preference exports, statistics, evidence, and the persisted `grid_axes` provenance (pinned selected doc ids / node units, mix, seed).

## Family Boundary

A family runtime must implement:

- `train_f(...)`
- `train_g(...)`
- `score_roots_with_f(...)`
- `validate_artifact(...)`

Built-in family names are intentionally few: `oracle`, `learnable_constant`, `classical_sketch`, `neural_operator`, `fno`, `llm`, and `dspy`. Register application workflows from the owning downstream package; `treepo` keeps its family registry small and branch-free.

When adding a built-in family, it should be dependency-light, generally useful, small enough for package tests, and registered in `treepo.methods.families`. Put family-specific knobs in `backend_config`; keep the public fit shape stable.

## Local-Law Boundary

`treepo.local_law` is canonical for scalar C1/C2/C3 row arithmetic: IPW, propensity clipping, depth weighting, and corrected local-law losses all live here, and examples and families route through it. Build `LocalLawAuditRow` values and call:

- `corrected_local_law_loss(...)`
- `local_law_objective_summary(...)`
- `audit_local_laws(...)`
- `compute_influence_weighted_overlap(...)`
- `triangle_local_law_residual_from_audit(...)`
- `build_triangle_local_law_error_certificate(...)`

Training code may wrap these operations in tensors, but theorem-facing rows and reports should round-trip through the dataclasses here.

For error estimation, the audited local-law objective is the Python estimator
for the leaf-up/triangle transport premise. Use
`triangle_local_law_residual_from_audit(...)` to turn an `audit_local_laws`
payload, or raw rows, into the leaf-up channel of a `TwoChannelResidual`; use
`build_triangle_local_law_error_certificate(...)` when a run also has
document-level root controls, overidentification residuals, common-mechanism
root-error envelopes, or external conditional-average envelopes. Propensities
remain sampling probabilities. Structural identification weights, including
additive qsentence/CMP weights, belong in `node_weight` and metadata.

## Preference And Fine-Tuning Boundary

`PreferenceDataset` is the storage boundary. Pairwise DPO, reward-model, GRPO, SFT, embedding pair/triplet, and ranked-row files are all projections of the same dataset.

Use:

- `PreferenceDataset.from_value(...)` to accept records, tables, paths, or compatible objects.
- `PreferenceDataset.to_records(...)` for `general`, `supervised`, `dpo`, `reward`, and `grpo` views.
- `treepo.finetune` for trainer-neutral embedding and LLM export views.

Examples export rows that downstream trainers consume; TRL, sentence-transformers, and serving stacks run downstream against the exported files.

## LLM And Server Boundary

`treepo.llm` provides client-side request/response helpers and optional embedding/chat clients. The deploying package owns server startup, GPU placement, vLLM/SGLang lifecycle, Transformers pipeline construction, large model downloads, and provider credentials.

The `llm` and `dspy` method families are provider-neutral. Use `api_base` for
OpenAI-compatible endpoints such as vLLM, SGLang, hosted compatible APIs, TGI,
or llama.cpp's OpenAI server. Use `predict_fn` for direct local runtimes such as
Hugging Face Transformers pipelines, custom Python inference functions, or SDKs
that do not expose `/v1`. DSPy uses an injected callable/program; DSPy and
model-serving libraries load only when downstream code imports or runs them.

## Examples Policy

Examples must be small, runnable, and package-native. They should use `treepo.fit(...)`, `PreferenceDataset`, `TreeRecord`, `TaskState`, local-law rows, and evidence artifacts. Publication grids live in downstream workspaces.

Good examples:

- `examples/bench/*.yaml` for `treepo-bench` runs.
- `examples/methods/*.toml` plus a small `run_*.py` wrapper.
- `examples/methods/run_llm_backends.py` for OpenAI-compatible, Transformers-style callable, and DSPy-style backend adapter shapes.
- Manifesto examples that use packaged fixtures and export trainer-neutral records.

Large real-data preparation, LLM scoring campaigns, and long runs belong in downstream workspaces.

## Dependency Hygiene

Top-level `import treepo` must stay light. Release checks reject heavy optional imports in core-light modules. Keep imports lazy for optional stacks such as `datasets`, DSPy, OpenAI, vLLM, torch, pandas, transformers, sentence-transformers, TRL, and PEFT.

Other release-hygiene rules:

- Keep generated artifacts, caches, logs, and local outputs out of the git tree.
- Use repo-relative paths in tracked docs/configs.
- Document environment setup through the `uv` workflow.
- Keep `inventory.yaml` aligned when adding or moving package areas.

## Change Patterns

| Task | Preferred move |
| --- | --- |
| Add a new small built-in family | Implement a `FamilyRuntime`, register it in `methods/families.py`, add focused tests and a tiny example if useful. |
| Add a large application family | Keep it outside `treepo`; register a factory from the downstream package. |
| Add task labels or structured summaries | Use `TaskState`, `TreeRecord`, and `PreferenceDataset`. |
| Add local-law supervision | Emit `LocalLawAuditRow`; summarize through `treepo.local_law`. |
| Add trainer data exports | Project `PreferenceDataset` through `treepo.finetune` or `PreferenceDataset.to_records(...)`. |
| Add a benchmark | Prefer `treepo.bench.tasks`; validate config keys and write JSON/CSV. |
| Add docs | Link from `README.md` if user-facing; use repo-relative paths and `uv` commands. |
| Add optional dependencies | Put them behind extras and lazy imports; update release tests if the public boundary changes. |

## Tests To Run

Use focused tests first, then broaden based on risk.

Common focused checks:

```bash
uv run pytest -q tests/test_package_layers.py tests/test_release_gates.py tests/test_unified_contracts.py
uv run pytest -q tests/methods/test_family_surface.py tests/methods/test_examples_smoke.py
uv run pytest -q tests/bench/test_markov_runner.py tests/sketches/test_broad_classical_sketches.py
```

Release boundary checks:

```bash
uv lock --check
uv run treepo-bench check release --json
uv run python -m treepo.release
uv build --wheel --sdist --out-dir /tmp/treepo_release_artifacts
```

Full test pass:

```bash
uv run pytest -q
```

## Review Checklist

Before handing work back:

- `git diff` only includes intentional files.
- Public imports remain lazy.
- `treepo.fit(...)` call shape is unchanged unless the user explicitly asked for a breaking change.
- New data uses existing package records rather than bespoke dicts.
- Optional/heavy dependencies are behind extras and local imports.
- Examples stay small, self-contained, and runnable from packaged fixtures.
- Release checks that cover the touched area have run, or failures are reported clearly.

## Existing Docs

- `README.md`: user-facing overview and quick start.
- `docs/architecture.md`: layer and experiment-contract overview.
- `docs/boundary.md`: package inclusion/exclusion policy.
- `docs/training_defaults.md`: fit defaults, built-in families, and extension boundary.
- `docs/methods_module_layout.md`: internal `treepo.methods` decomposition convention.
- `docs/evidence.md`: the per-run evidence artifact shape and semantics.
